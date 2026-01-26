import numpy as np
import torch
from torch import nn
from sklearn.base import BaseEstimator
from .km import KM


class PUe(BaseEstimator):
    def __init__(self, lr: float = 1e-3, epochs: int = 20):
        self.lr = lr
        self.epochs = epochs
        self.e = None
        self.clf = None
        self.alpha = 15  # alpha=15 as recommended in the original paper

    def fit(self, X, s):
        """
        Fits the PUe model to the data.

        Parameters
        ----------
        X : numpy.ndarray
            The data to fit the model to.
        s : numpy.ndarray
            The observed labels of the data.
        """
        # Estimate class prior
        X_mixture = X[np.where(s == 0)[0], :]
        X_component = X[np.where(s == 1)[0], :]
        km_estimator = KM()
        est = km_estimator.estimate(X_mixture, X_component)
        est_pi = (1 - np.mean(s)) * est["km2"] + np.mean(s)

        X = torch.from_numpy(X).float()
        s = torch.from_numpy(s).float()
        self.e = nn.Linear(X.shape[1], 1)
        criterion = LossE(
            n_p=torch.sum(s), n_U=s.shape[0] - torch.sum(s), alpha=self.alpha
        )
        optimizer = torch.optim.Adam(self.e.parameters(), lr=1e-3)

        for epoch in range(20):
            optimizer.zero_grad()
            outputs = self.e(X)
            loss = criterion(outputs, s)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            prop_scores = torch.sigmoid(self.e(X))
        normalized_prop_scores = (
            torch.sum(torch.where(s == 1, 1 / prop_scores, 0)) * prop_scores
        )

        self.clf = nn.Linear(X.shape[1], 1)
        criterion = LossCLF(pi=est_pi)
        optimizer = torch.optim.Adam(self.clf.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.clf(X)
            loss = criterion(outputs, s, normalized_prop_scores)
            loss.backward()
            optimizer.step()

    def predict(self, X):
        """
        Parameters
        ----------
        X : numpy.ndarray
            The data to predict the labels of.
        """
        return np.where(self.predict_proba(X) > 0.5, 1, 0)

    def predict_proba(self, X):
        """
        Parameters
        ----------
        X : numpy.ndarray
            The data to predict the probabilities of."""
        with torch.no_grad():
            X = torch.from_numpy(X).float()
            scores = self.clf(X)
            probs = torch.sigmoid(scores).squeeze().numpy()
            return np.array(list(zip(1 - probs, probs)))


class LossE(nn.Module):
    def __init__(self, n_p, n_U, alpha) -> None:
        """
        Parameters
        ----------
        n_p : int
            The number of positive samples.
        n_U : int
            The number of unlabeled samples.
        alpha : float
            The alpha parameter of the model.
        """
        super().__init__()
        self.n_p = n_p
        self.n_U = n_U
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        """
        Calculates the loss of the model.

        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted values of the model.
        y_true : torch.Tensor
            The true values of the model.
        """
        loss1 = torch.sum(
            -1 / (self.n_p + self.n_U) * y_true * torch.log(torch.sigmoid(y_pred))
        )
        loss2 = torch.sum(
            -1
            / (self.n_p + self.n_U)
            * (1 - y_true)
            * torch.log(1 - torch.sigmoid(y_pred))
        )
        regularisation = self.alpha * torch.abs(
            torch.sum(torch.sigmoid(y_pred)) - self.n_p
        )
        loss = loss1 + loss2 + regularisation
        return loss


class LossCLF(nn.Module):
    def __init__(self, pi) -> None:
        """
        Parameters
        ----------
        pi : float
            The class prior of the data.
        """
        super().__init__()
        self.pi = pi

    def forward(self, y_pred, y_true, prop_scores):
        """
        Calculates the loss of the model.

        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted values of the model.
        y_true : torch.Tensor
            The true values of the model.
        prop_scores : torch.Tensor
            The prop scores of the model.
        """
        loss1 = torch.sum(-1 / prop_scores * y_true * torch.log(torch.sigmoid(y_pred)))
        loss2 = torch.sum(-(1 - y_true) * torch.log(1 - torch.sigmoid(y_pred)))
        loss3 = torch.sum(
            -1 / prop_scores * y_true * torch.log(1 - torch.sigmoid(y_pred))
        )

        return self.pi * loss1 + loss2 - self.pi * loss3
