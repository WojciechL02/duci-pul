import numpy as np
from sklearn.base import BaseEstimator
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


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
        pretraining_epochs: int = 50,
        epochs: int = 50,
        m_steps: int = 25,
        pretraining_lr: float = 1e-2,
        lr: float = 1e-4,
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
