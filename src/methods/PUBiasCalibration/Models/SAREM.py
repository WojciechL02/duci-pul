import numpy as np
import torch
import tqdm

from torch import nn
from ..helper_files.classifiers import MLPReLU, LR, FullCNN, Resnet
from ..helper_files.sarpu.pu_learning import pu_learn_sar_em
from ..helper_files.utils import EarlyStopping
from copy import deepcopy
from sklearn.base import BaseEstimator


def seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


class SAREM(BaseEstimator):
    def __init__(self):
        self.model = None

    def fit(self, X, s):
        """
        Fits the threshold model to the data.

        Parameters
        ----------
        X : numpy.ndarray
            The data to fit the model to.
        s : numpy.ndarray
            The observed labels of the data.
        """
        model, _, _ = pu_learn_sar_em(X, s, range(X.shape[1]))
        self.model = model

    def predict_proba(self, X):
        """
        Predicts the probability of the data being positive.

        Parameters
        ----------
        X : numpy.ndarray
            The data to predict the probability of.
        """
        y_pred = self.model.predict_proba(X)
        return np.column_stack((1 - y_pred, y_pred))


class SAREMdeep(nn.Module):
    def __init__(self, clf, dims, device=0) -> None:
        """
        Initializes the deep SAREM model.

        Parameters
        ----------
        clf : str
            The classifier to use. Options are 'lr', 'mlp', 'cnn', 'resnet'.
        dims : list
            The dimensions of the data.
        device : int
            The device to use for training.
        """
        super().__init__()
        self.device = (
            "mps"
            if getattr(torch, "has_mps", False)
            else "cuda:{}".format(device) if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")

        if clf == "lr" or clf == "mlp":
            assert (
                dims != None
            ), "Classifier type {} requires specifying the dimensionality of the data.".format(
                clf
            )

        if clf == "lr":
            self.f = LR(dims=dims).to(self.device)
            self.e = LR(dims=dims).to(self.device)
        elif clf == "mlp":
            self.f = MLPReLU(dims=dims).to(self.device)
            self.e = MLPReLU(dims=dims).to(self.device)
        elif clf == "cnn":
            self.f = FullCNN().to(self.device)
            self.e = FullCNN().to(self.device)
        elif clf == "resnet":
            self.f = Resnet().to(self.device)
            self.e = Resnet().to(self.device)

    def predict_proba(self, x):
        """
        Predicts the probability of the data being positive.

        Parameters
        ----------
        x : torch.Tensor
            The data to predict the probability of.
        """
        with torch.no_grad():
            return self.f(x, probabilistic=True)

    def initialise_f(self, trainloader, valloader, epochs=100, lr=1e-3):
        """
        Initialises the classifier f for the EM algorithm by fitting a balanced model which considers the unlabeled examples as negatives.

        Parameters
        ----------
        trainloader : torch.utils.data.DataLoader
            The data to train the model on.
        valloader : torch.utils.data.DataLoader
            The data to validate the model on.
        epochs : int
            The number of epochs to train the model for.
        lr : float
            The learning rate to use for training.
        """
        proportion_labeled = 0
        for _, labels in trainloader:
            labels = labels.to(self.device)
            proportion_labeled += torch.sum(labels)
        proportion_labeled = proportion_labeled / len(trainloader.dataset)

        es = EarlyStopping()

        optimizer = torch.optim.Adam(self.f.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        done = False

        for epoch in range(epochs):
            steps = list(enumerate(trainloader))
            pbar = tqdm.tqdm(steps)
            for i, data in pbar:

                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                optimizer.zero_grad()

                outputs = self.f(inputs, probabilistic=False)
                loss_unweighted = criterion(
                    outputs, labels.unsqueeze(1).float()
                ).squeeze()
                weights = torch.where(
                    labels == 1, (1 - proportion_labeled), proportion_labeled
                )
                loss_weighted = torch.mean(loss_unweighted * weights)
                loss_weighted.backward()
                optimizer.step()

                loss = loss_weighted.item()
                if i == len(steps) - 1:
                    self.eval()
                    v_loss = 0
                    for j, val_data in enumerate(valloader):
                        with torch.no_grad():
                            inputs, labels = val_data[0].to(self.device), val_data[
                                1
                            ].to(self.device)
                            pred_y = self.f(inputs, probabilistic=False)
                            loss_unweighted = criterion(
                                pred_y, labels.unsqueeze(1).float()
                            ).squeeze()
                            weights = torch.where(
                                labels == 1,
                                (1 - proportion_labeled),
                                proportion_labeled,
                            )
                            v_loss += torch.mean(loss_unweighted * weights).item()
                    v_loss = v_loss / (j + 1)

                    if es(self.f, v_loss):
                        done = True
                    pbar.set_description(
                        f"Initialising classifier - Epoch: {epoch}, tloss: {loss}, vloss: {v_loss:>7f}, EStop:[{es.status}]"
                    )
                    self.train()
                else:
                    pbar.set_description(
                        f"Initialising classifier - Epoch: {epoch}, tloss {loss:}"
                    )
            if done == True:
                break

        self.f_frozen = deepcopy(self.f)

    def initialise_e(self, trainloader, valloader, epochs=100, lr=1e-3):
        """
        The propensity score model is initialized by using the classification model to estimate the probabilities of each unlabeled example being positive.

        Parameters
        ----------
        trainloader : torch.utils.data.DataLoader
            The data to train the model on.
        valloader : torch.utils.data.DataLoader
            The data to validate the model on.
        epochs : int
            The number of epochs to train the model for.
        lr : float
            The learning rate to use for training.
        """
        es = EarlyStopping()

        optimizer = torch.optim.Adam(self.e.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        done = False
        for epoch in range(epochs):
            steps = list(enumerate(trainloader))
            pbar = tqdm.tqdm(steps)
            for i, data in pbar:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # We set the mode of f to evaluation, which is put back to training after early stopping with self.train()
                self.f.eval()
                with torch.no_grad():
                    f_pred = self.f(inputs, probabilistic=True)

                optimizer.zero_grad()
                outputs = self.e(inputs, probabilistic=False)
                loss_unweighted = criterion(
                    outputs, labels.unsqueeze(1).float()
                ).squeeze()
                weights = torch.where(labels == 1, 1, f_pred)
                loss_weighted = torch.mean(loss_unweighted * weights)
                loss_weighted.backward()
                optimizer.step()

                loss = loss_weighted.item()
                if i == len(steps) - 1:
                    self.eval()
                    v_loss = 0
                    with torch.no_grad():
                        for j, val_data in enumerate(valloader):
                            inputs, labels = val_data[0].to(self.device), val_data[
                                1
                            ].to(self.device)
                            f_pred = self.f(inputs, probabilistic=True)
                            pred_y = self.e(inputs, probabilistic=False)
                            loss_unweighted = criterion(
                                pred_y, labels.unsqueeze(1).float()
                            ).squeeze()
                            weights = torch.where(labels == 1, 1, f_pred)
                            v_loss += torch.mean(loss_unweighted * weights).item()
                    v_loss = v_loss / (j + 1)

                    if es(self.e, v_loss):
                        done = True
                    pbar.set_description(
                        f"Initialising propensity score - Epoch: {epoch}, tloss: {loss}, vloss: {v_loss:>7f}, EStop:[{es.status}]"
                    )
                    self.train()
                else:
                    pbar.set_description(
                        f"Initialising propensity score - Epoch: {epoch}, tloss {loss:}"
                    )
            if done == True:
                break

        self.e_frozen = deepcopy(self.e)

    def fit(self, trainloader, valloader, epochs=100, lr=1e-3):
        """
        Fits the SAREM model to the data using EM.

        Parameters
        ----------
        trainloader : torch.utils.data.DataLoader
            The data to train the model on.
        valloader : torch.utils.data.DataLoader
            The data to validate the model on.
        epochs : int
            The number of epochs to train the model for.
        lr : float
            The learning rate to use for training.
        """
        self.initialise_f(trainloader, valloader, epochs, lr)
        self.initialise_e(trainloader, valloader, epochs, lr)

        es = EarlyStopping()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        done = False
        for epoch in range(epochs):
            steps = list(enumerate(trainloader))
            pbar = tqdm.tqdm(steps)
            for i, data in pbar:

                inputs, s = data[0].to(self.device), data[1].to(self.device)
                optimizer.zero_grad()

                # Expectation step
                self.eval()
                with torch.no_grad():
                    f_x = self.f_frozen(inputs, probabilistic=True).squeeze()
                    e_x = self.e_frozen(inputs, probabilistic=True).squeeze()
                    y_hat = s + (1 - s) * f_x * (1 - e_x) / (1 - f_x * e_x)
                self.train()

                # Maximization step
                f_loss = torch.mean(
                    criterion(self.f(inputs, probabilistic=False), y_hat.unsqueeze(1))
                )
                e_loss = torch.mean(
                    y_hat
                    * criterion(
                        self.e(inputs, probabilistic=False).squeeze(), s.float()
                    )
                )
                loss = f_loss + e_loss
                loss.backward()
                optimizer.step()

                loss = loss.item()
                if i == len(steps) - 1:
                    # Early stopping check
                    self.eval()
                    v_loss = 0
                    with torch.no_grad():
                        for j, val_data in enumerate(valloader):
                            inputs, s = data[0].to(self.device), data[1].to(self.device)

                            f_x = self.f_frozen(inputs, probabilistic=True).squeeze()
                            e_x = self.e_frozen(inputs, probabilistic=True).squeeze()
                            y_hat = s + (1 - s) * f_x * (1 - e_x) / (1 - f_x * e_x)

                            f_loss = torch.mean(
                                criterion(
                                    self.f(inputs, probabilistic=False),
                                    y_hat.unsqueeze(1),
                                )
                            ).item()
                            e_loss = torch.mean(
                                y_hat
                                * criterion(
                                    self.e(inputs, probabilistic=False).squeeze(),
                                    s.float(),
                                )
                            ).item()

                            v_loss += f_loss + e_loss
                    v_loss = v_loss / (j + 1)
                    if es(self.f, v_loss):
                        done = True
                    pbar.set_description(
                        f"EM - Epoch: {epoch}, tloss: {loss}, vloss: {v_loss:>7f}, EStop:[{es.status}]"
                    )
                    self.train()
                else:
                    pbar.set_description(f"EM - Epoch: {epoch}, tloss {loss:}")
            if done == True:
                break
            # f_frozen and e_frozen are updated only after the epoch is done, as to not use updated models in the expectation step.
            self.f_frozen = deepcopy(self.f)
            self.e_frozen = deepcopy(self.e)
