import torch
import tqdm
import numpy as np
from ..helper_files import km

from torch import nn
from ..helper_files.classifiers import LR, MLPReLU, FullCNN, Resnet
from ..helper_files.utils import EarlyStopping
from sklearn.base import BaseEstimator


class CustomLoss_e(nn.Module):
    def __init__(self, n_p, n_U, alpha) -> None:
        """
        Initializes the custom loss function for the e model.

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


class CustomLoss_clf(nn.Module):
    def __init__(self, pi) -> None:
        """
        Initializes the custom loss function for the classifier model.

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


# Model definition
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


class PUe(BaseEstimator):

    def __init__(self):
        """
        Initializes the PUe model.
        """
        self.e = None
        self.clf = None

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
        km1 = km.km_default(X_mixture, X_component)
        est_pi = (1 - np.mean(s)) * km1[1] + np.mean(s)

        X = torch.from_numpy(X).float()
        s = torch.from_numpy(s).float()
        self.e = LogisticRegressionModel(X.shape[1])
        criterion = CustomLoss_e(
            n_p=torch.sum(s), n_U=s.shape[0] - torch.sum(s), alpha=15
        )  # Alpha=15 as recommended in the original paper.
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

        self.clf = LogisticRegressionModel(X.shape[1])
        criterion = CustomLoss_clf(pi=est_pi)
        optimizer = torch.optim.Adam(self.clf.parameters(), lr=1e-3)

        for epoch in range(20):
            optimizer.zero_grad()
            outputs = self.clf(X)
            loss = criterion(outputs, s, normalized_prop_scores)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, X):
        """
        Predicts the labels of the data.

        Parameters
        ----------
        X : numpy.ndarray
            The data to predict the labels of.
        """
        return np.where(self.predict_proba(X) > 0.5, 1, 0)

    def predict_proba(self, Xtest):
        """
        Predicts the probabilities of the data.

        Parameters
        ----------
        Xtest : numpy.ndarray
            The data to predict the probabilities of."""
        with torch.no_grad():
            Xtest = torch.from_numpy(Xtest).float()
            scores = self.clf(Xtest)
            probs = torch.sigmoid(scores).squeeze().numpy()
            return np.array(list(zip(1 - probs, probs)))


class PUedeep(nn.Module):
    def __init__(self, clf, dims=None, est_pi=None, device=0) -> None:
        """
        Initializes the PUedeep model.

        Parameters
        ----------
        clf : str
            The type of classifier to use.
        dims : tuple
            The dimensions of the data.
        est_pi : float
            The class prior of the data.
        device : int
            The device to use.
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
            self.e = LR(dims=dims).to(self.device)
            self.clf = LR(dims=dims).to(self.device)
        elif clf == "mlp":
            self.e = MLPReLU(dims=dims).to(self.device)
            self.clf = MLPReLU(dims=dims).to(self.device)
        elif clf == "cnn":
            self.e = FullCNN().to(self.device)
            self.clf = FullCNN().to(self.device)
        elif clf == "resnet":
            self.e = Resnet().to(self.device)
            self.clf = Resnet().to(self.device)

        self.pi = est_pi

    def predict_proba(self, x):
        """
        Predicts the probabilities of the data.

        Parameters
        ----------
        x : torch.Tensor
            The data to predict the probabilities of."""
        with torch.no_grad():
            scores = self.clf(x, probabilistic=False)
            return torch.sigmoid(scores)

    def calculate_prop_scores(self, trainloader):
        """
        Calculates the prop scores of the data and set the normalizing factor.

        Parameters
        ----------
        trainloader : torch.utils.data.DataLoader
            The data to calculate the prop scores of.
        """
        with torch.no_grad():
            prop_scores = []
            s = []
            for i, data in enumerate(trainloader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                pred_y = torch.sigmoid(self.e(inputs, probabilistic=False))
                prop_scores.append(pred_y)
                s.append(labels)
            prop_scores = torch.cat(prop_scores)
            s = torch.cat(s)

        self.normalizing_factor = torch.sum(torch.where(s == 1, 1 / prop_scores, 0))

    def fit(self, trainloader, valloader, epochs, lr=1e-3):
        """
        Fits the PUedeep model to the data.

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
        optimizer_e = torch.optim.Adam(self.e.parameters(), lr=lr)
        optimizer_clf = torch.optim.Adam(self.clf.parameters(), lr=lr)

        count = 0
        count_positive = 0
        for data, labels in trainloader:
            count_positive += torch.sum(labels)
            count += len(labels)

        criterion_e = CustomLoss_e(
            n_p=count_positive, n_U=count - count_positive, alpha=15
        )  # Alpha=15 as recommended in the original paper.
        criterion_clf = CustomLoss_clf(self.pi)

        es = EarlyStopping()

        # Fit the e model
        done = False
        for epoch in range(epochs):
            steps = list(enumerate(trainloader))
            pbar = tqdm.tqdm(steps)
            for i, data in pbar:

                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                optimizer_e.zero_grad()
                outputs = self.e(inputs, probabilistic=False)
                loss = criterion_e(outputs, labels.unsqueeze(1).float())
                loss.backward()
                optimizer_e.step()

                loss = loss.item()
                if i == len(steps) - 1:
                    v_loss = 0

                    for j, val_data in enumerate(valloader):
                        inputs, labels = val_data[0].to(self.device), val_data[1].to(
                            self.device
                        )
                        pred_y = self.e(inputs, probabilistic=False)
                        v_loss += criterion_e(
                            pred_y, labels.unsqueeze(1).float()
                        ).item()

                    v_loss = v_loss / (j + 1)

                    if es(self.e, v_loss):
                        done = True

                    pbar.set_description(
                        f"Epoch: {epoch}, tloss: {loss}, vloss: {v_loss:>7f}, EStop:[{es.status}]"
                    )

                else:
                    pbar.set_description(f"Epoch: {epoch}, tloss: {loss:}")

            if done == True:
                break

        self.calculate_prop_scores(trainloader)

        es = EarlyStopping()

        # Fit the clf model
        done = False
        for epoch in range(epochs):
            steps = list(enumerate(trainloader))
            pbar = tqdm.tqdm(steps)
            for i, data in pbar:

                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                with torch.no_grad():
                    prop_scores = torch.sigmoid(self.e(inputs, probabilistic=False))
                    normalized_prop_scores = self.normalizing_factor * prop_scores
                optimizer_clf.zero_grad()
                outputs = self.clf(inputs, probabilistic=False)
                loss = criterion_clf(
                    outputs, labels.unsqueeze(1).float(), normalized_prop_scores
                )
                loss.backward()
                optimizer_clf.step()

                loss = loss.item()
                if i == len(steps) - 1:
                    v_loss = 0

                    for j, val_data in enumerate(valloader):
                        inputs, labels = val_data[0].to(self.device), val_data[1].to(
                            self.device
                        )
                        with torch.no_grad():
                            prop_scores = torch.sigmoid(
                                self.e(inputs, probabilistic=False)
                            )
                            normalized_prop_scores = (
                                self.normalizing_factor * prop_scores
                            )
                        pred_y = self.clf(inputs, probabilistic=False)
                        v_loss += criterion_clf(
                            pred_y, labels.unsqueeze(1).float(), normalized_prop_scores
                        ).item()

                    v_loss = v_loss / (j + 1)

                    if es(self.clf, v_loss):
                        done = True

                    pbar.set_description(
                        f"Epoch: {epoch}, tloss: {loss}, vloss: {v_loss:>7f}, EStop:[{es.status}]"
                    )

                else:
                    pbar.set_description(f"Epoch: {epoch}, tloss: {loss:}")

            if done == True:
                break
