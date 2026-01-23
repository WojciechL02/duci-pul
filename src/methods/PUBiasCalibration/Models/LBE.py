import torch
import tqdm
from copy import deepcopy
import numpy as np
from sklearn.base import BaseEstimator
from torch import nn
from torch import optim
import torch.nn.functional as F
from ..helper_files.classifiers import MLPReLU, FullCNN, LR, Resnet
from ..helper_files.utils import EarlyStopping


class Classifier(nn.Module):
    def forward(self, x):
        pass


class MLPClassifier(Classifier):
    def __init__(self, input_dim, hidden_dim=10):
        super(MLPClassifier, self).__init__()
        self.h = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            # nn.Sigmoid(),
        )

    def forward(self, x, sigmoid=False):
        h = self.h(x)
        if sigmoid:
            h = torch.sigmoid(h)
        return h


class LogisticClassifier(Classifier):
    def __init__(self, input_dim):
        super(LogisticClassifier, self).__init__()
        self.h = nn.Sequential(
            nn.Linear(input_dim, 1),
            # nn.Sigmoid(),
        )

    def forward(self, x, sigmoid=False):
        h = self.h(x)
        if sigmoid:
            h = torch.sigmoid(h)
        return h


class PropensityEstimator(nn.Module):
    def forward(self, x):
        pass


class MLPPropensityEstimator(PropensityEstimator):
    def __init__(self, input_dim, hidden_dim=10):
        super(MLPPropensityEstimator, self).__init__()
        self.eta = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            # nn.Sigmoid(),
        )

        for layer in [
            module for module in self.eta.modules() if isinstance(module, nn.Linear)
        ]:
            # layer.weight = nn.Parameter(torch.randn_like(layer.weight) / 100)
            # layer.bias = nn.Parameter(torch.randn_like(layer.bias) / 100)
            layer.weight = nn.Parameter(torch.zeros_like(layer.weight))
            layer.bias = nn.Parameter(torch.zeros_like(layer.bias))

    def forward(self, x, sigmoid=False):
        eta = self.eta(x)
        if sigmoid:
            eta = torch.sigmoid(eta)
        return eta


class LogisticPropensityEstimator(PropensityEstimator):
    def __init__(self, input_dim):
        super(LogisticPropensityEstimator, self).__init__()
        self.eta = nn.Sequential(
            nn.Linear(input_dim, 1),
            # nn.Sigmoid(),
        )

        for layer in [
            module for module in self.eta.modules() if isinstance(module, nn.Linear)
        ]:
            # layer.weight = nn.Parameter(torch.randn_like(layer.weight) / 100)
            # layer.bias = nn.Parameter(torch.randn_like(layer.bias) / 100)
            layer.weight = nn.Parameter(torch.zeros_like(layer.weight))
            layer.bias = nn.Parameter(torch.zeros_like(layer.bias))

    def forward(self, x, sigmoid=False):
        eta = self.eta(x)
        if sigmoid:
            eta = torch.sigmoid(eta)
        return eta


class LBENetwork(nn.Module):
    def __init__(self, input_dim, kind="LR", device=None):
        super(LBENetwork, self).__init__()
        self.device = device
        hidden_dim = 2 * input_dim
        if kind == "MLP":
            self.h = MLPClassifier(input_dim, hidden_dim)
            self.eta = MLPPropensityEstimator(input_dim, hidden_dim)
        elif kind == "LR":
            self.h = LogisticClassifier(input_dim)
            self.eta = LogisticPropensityEstimator(input_dim)

    def get_theta_h(self):
        value = torch.tensor([], dtype=torch.float, device=self.device)
        for param in self.h.parameters():
            value = torch.cat([value.squeeze(), param.data.reshape(-1)])
        return value

    def get_theta_eta(self):
        value = torch.tensor([], dtype=torch.float, device=self.device)
        for param in self.eta.parameters():
            value = torch.cat([value.squeeze(), param.data.reshape(-1)])
        return value

    def forward(self, x):
        x = x.to(self.device)
        h = self.h(x, sigmoid=True)
        return h

    def E_step(self, x, s):
        with torch.no_grad():
            x = x.to(self.device)
            s = s.to(self.device)

            h = self.h(x, sigmoid=True).squeeze()
            eta = self.eta(x, sigmoid=True).squeeze()

            P_y_hat_1 = torch.where(s == 1, eta, 1 - eta) * h
            P_y_hat_0 = torch.where(s == 1, 0, 1) * (1 - h)

            P_y_hat = torch.cat(
                [P_y_hat_0.reshape(-1, 1), P_y_hat_1.reshape(-1, 1)], axis=1
            )
            P_y_hat /= P_y_hat.sum(axis=1).reshape(-1, 1)
            return P_y_hat

    def loss(self, x, s, P_y_hat):
        x = x.to(self.device)
        s = s.to(self.device)
        P_y_hat = P_y_hat.to(self.device)

        h = self.h(x).squeeze()
        eta = self.eta(x).squeeze()

        log_h = F.logsigmoid(h)
        log_1_minus_h = F.logsigmoid(-h)
        log_eta = F.logsigmoid(eta)
        log_1_minus_eta = F.logsigmoid(-eta)

        loss = torch.where(
            s == 1,
            P_y_hat[:, 1] * (log_h + log_eta)
            + P_y_hat[:, 0] * (log_1_minus_h + log_eta),
            P_y_hat[:, 1] * (log_h + log_1_minus_eta)
            + P_y_hat[:, 0] * (log_1_minus_h + log_1_minus_eta),
        )
        with torch.no_grad():
            x_ones = torch.ones((len(x), 1), device=self.device)
            x = torch.cat([x, x_ones], dim=1)
            sigma_h = torch.sigmoid(h)
            sigma_eta = torch.sigmoid(eta)
            grad_theta_1 = (
                (P_y_hat[:, 0] * sigma_h).reshape(-1, 1) * x
                + (P_y_hat[:, 1] * (sigma_h - 1)).reshape(-1, 1) * x
            ).sum(axis=0)
            grad_theta_2 = (
                (
                    (-1) ** (s + 1)
                    * (
                        1
                        * P_y_hat[:, 1]
                        / torch.where(s == 1, sigma_eta, 1 - sigma_eta)
                    )
                    * sigma_eta
                    * (sigma_eta - 1)
                ).reshape(-1, 1)
                * x
            ).sum(axis=0)

        return -torch.sum(loss), grad_theta_1, grad_theta_2

    def pre_train(self, x, s, epochs=100, lr=1e-3):
        x = x.to(self.device)
        s = s.to(self.device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.h.parameters(), lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            s_logits = self.h(x)
            loss = criterion(s_logits.squeeze(), s)
            loss.backward()
            optimizer.step()


class LBE(BaseEstimator):
    def __init__(
        self,
        pretraining_epochs: int,
        epochs: int,
        m_steps: int,
        pretraining_lr: float,
        lr: float,
        kind: str = "LR",
    ) -> None:
        self.model = None
        self.pretraining_epochs = pretraining_epochs
        self.epochs = epochs
        self.m_steps = m_steps
        self.pretraining_lr = pretraining_lr
        self.lr = lr
        self.kind = kind
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X: np.ndarray, s: np.ndarray) -> None:
        """
        Parameters
        ----------
        X : numpy.ndarray
            The data to fit the model to.
        s : numpy.ndarray
            The observed labels of the data.
        """
        p = X.shape[1]
        self.model = LBENetwork(p, kind=self.kind, device=self.device)
        self.model.to(self.device)
        X = torch.from_numpy(X).float()
        s = torch.from_numpy(s).float()

        # Pre-training
        self.model.train()
        self.model.pre_train(
            X, s, epochs=self.pretraining_epochs, lr=self.pretraining_lr
        )

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        for epoch in range(self.epochs):
            p_y_hat = self.model.E_step(X, s)

            for M_step_iter in range(self.m_steps):
                optimizer.zero_grad()
                loss, grad_theta_1, grad_theta_2 = self.model.loss(X, s, p_y_hat)
                loss.backward()
                optimizer.step()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the probabilities of the data.

        Parameters
        ----------
        X : numpy.ndarray
            The data to predict the probabilities of.
        """
        if self.model is None:
            raise RuntimeError("Model not trained, call fit() first!")

        self.model.eval()
        with torch.no_grad():
            X_data = torch.from_numpy(X).float()
            outputs = self.model(X_data)
            y_proba = outputs.squeeze().detach().cpu().numpy()
            return np.column_stack((1 - y_proba, y_proba))


class LBEdeep(nn.Module):
    def __init__(self, clf, dims=None, device=0):
        """
        Initializes the LBE model.

        Parameters
        ----------
        clf : str
            The type of classifier to use.
        dims : list
            The dimensions of the data.
        device : int
            The device to use.
        """
        super().__init__()
        self.device = "cuda:{}".format(device) if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        if clf == "lr" or clf == "mlp":
            assert (
                dims != None
            ), "Classifier type {} requires specifying the dimensionality of the data.".format(
                clf
            )

        if clf == "lr":
            self.h = LR(dims=dims).to(self.device)
            self.eta = LR(dims=dims).to(self.device)
        elif clf == "mlp":
            self.h = MLPReLU(dims=dims).to(self.device)
            self.eta = MLPReLU(dims=dims).to(self.device)
        elif clf == "cnn":
            self.h = FullCNN().to(self.device)
            self.eta = FullCNN().to(self.device)
        elif clf == "resnet":
            self.h = Resnet().to(self.device)
            self.eta = Resnet().to(self.device)

        self.h_frozen = deepcopy(self.h).to(self.device)
        self.eta_frozen = deepcopy(self.eta).to(self.device)

    def predict_proba(self, x):
        """
        Predicts the probabilities of the data.

        Parameters
        ----------
        x : torch.Tensor
            The data to predict the probabilities of.
        """
        with torch.no_grad():
            h = self.h(x, probabilistic=True)
            return h

    def E_step(self, x, s):
        """
        The E-step of the EM algorithm.

        Parameters
        ----------
        x : torch.Tensor
            The data to predict the probabilities of.
        s : torch.Tensor
            The observed labels of the data.
        """
        with torch.no_grad():
            h = self.h_frozen(x, probabilistic=True).squeeze()
            eta = self.eta_frozen(x, probabilistic=True).squeeze()

            P_y_hat_1 = torch.where(s == 1, eta, 1 - eta) * h
            P_y_hat_0 = torch.where(s == 1, 0, 1) * (1 - h)

            P_y_hat = torch.cat(
                [P_y_hat_0.reshape(-1, 1), P_y_hat_1.reshape(-1, 1)], axis=1
            )
            P_y_hat /= P_y_hat.sum(axis=1).reshape(-1, 1)
            return P_y_hat

    def loss(self, x, s, P_y_hat):
        """
        The loss function of the model.

        Parameters
        ----------
        x : torch.Tensor
            The data to predict the probabilities of.
        s : torch.Tensor
            The observed labels of the data.
        P_y_hat : torch.Tensor
            The predicted probabilities of the data.
        """
        h = self.h(x, probabilistic=False).squeeze()
        eta = self.eta(x, probabilistic=False).squeeze()

        log_h = F.logsigmoid(h)
        log_1_minus_h = F.logsigmoid(-h)
        log_eta = F.logsigmoid(eta)
        log_1_minus_eta = F.logsigmoid(-eta)

        loss1 = torch.where(
            s == 1,
            P_y_hat[:, 1] * log_eta + P_y_hat[:, 0] * log_eta,
            P_y_hat[:, 1] * log_1_minus_eta + P_y_hat[:, 0] * log_1_minus_eta,
        )

        loss2 = torch.where(
            s == 1,
            P_y_hat[:, 1] * log_h + P_y_hat[:, 0] * log_1_minus_h,
            P_y_hat[:, 1] * log_h + P_y_hat[:, 0] * log_1_minus_h,
        )

        loss = loss1 + loss2

        return -torch.sum(loss)

    def pre_fit(self, trainloader, valloader, epochs=100, lr=1e-3):
        """
        Initializes the classifier.

        Parameters
        ----------
        trainloader : torch.utils.data.DataLoader
            The data to fit the model to.
        valloader : torch.utils.data.DataLoader
            The data to validate the model on.
        epochs : int
            The number of epochs to train the model for.
        lr : float
            The learning rate of the model.
        """
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.h.parameters(), lr=lr)

        es = EarlyStopping()

        done = False
        for epoch in range(epochs):
            steps = list(enumerate(trainloader))
            pbar = tqdm.tqdm(steps)
            for i, data in pbar:

                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                optimizer.zero_grad()

                outputs = self.h(inputs, probabilistic=False)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                loss.backward()
                optimizer.step()

                loss = loss.item()
                if i == len(steps) - 1:
                    self.eval()
                    v_loss = 0
                    with torch.no_grad():
                        for j, val_data in enumerate(valloader):
                            inputs, labels = val_data[0].to(self.device), val_data[
                                1
                            ].to(self.device)
                            pred_y = self.h(inputs, probabilistic=False)
                            v_loss += criterion(
                                pred_y, labels.unsqueeze(1).float()
                            ).item()
                    v_loss = v_loss / (j + 1)
                    if es(self.h, v_loss):
                        done = True
                    pbar.set_description(
                        f"Pre-training - Epoch: {epoch}, tloss: {loss}, vloss: {v_loss:>7f}, EStop:[{es.status}]"
                    )
                    self.train()
                else:
                    pbar.set_description(
                        f"Pre-training - Epoch: {epoch}, tloss {loss:}"
                    )
            if done == True:
                break
        self.h_frozen = deepcopy(self.h)

    def fit(self, trainloader, valloader, epochs=100, lr=1e-3):
        """
        Applies EM to the propensity scores eta and classifier h.

        Parameters
        ----------
        trainloader : torch.utils.data.DataLoader
            The data to fit the model to.
        valloader : torch.utils.data.DataLoader
            The data to validate the model on.
        epochs : int
            The number of epochs to train the model for.
        lr : float
            The learning rate of the model.
        """
        self.pre_fit(trainloader, valloader, epochs=epochs, lr=lr)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        es = EarlyStopping()

        for epoch in range(epochs):

            done = False

            steps = list(enumerate(trainloader))
            pbar = tqdm.tqdm(steps)
            for i, data in pbar:

                inputs, s = data[0].to(self.device), data[1].to(self.device)
                P_y_hat = self.E_step(inputs, s)

                optimizer.zero_grad()

                loss = self.loss(inputs, s, P_y_hat)

                loss.backward()
                optimizer.step()

                loss = loss.item()

                if i == len(steps) - 1:
                    self.eval()
                    v_loss = 0
                    with torch.no_grad():
                        for j, val_data in enumerate(valloader):
                            inputs, s = val_data[0].to(self.device), val_data[1].to(
                                self.device
                            )
                            P_y_hat = self.E_step(inputs, s)
                            v_loss = self.loss(inputs, s, P_y_hat)
                    v_loss = v_loss / (j + 1)
                    if es(self.h, v_loss):
                        done = True
                    pbar.set_description(
                        f"EM - Epoch: {epoch}, tloss: {loss}, vloss: {v_loss:>7f}, EStop:[{es.status}]"
                    )
                    self.train()
                else:
                    pbar.set_description(f"EM - Epoch: {epoch}, tloss {loss:}")

            self.h_frozen = deepcopy(self.h)
            self.eta_frozen = deepcopy(self.eta)

            if done == True:
                break
