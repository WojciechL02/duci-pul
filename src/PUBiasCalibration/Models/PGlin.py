import numpy as np
import torch
import tqdm
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from torch import nn

from ..helper_files.classifiers import LR, MLPReLU, FullCNN, Resnet
from ..helper_files.utils import EarlyStopping


def seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

class PUGerych(BaseEstimator):
    def __init__(self):
        """
        Initializes the PGlin model.
        """
        base_estimator = LogisticRegression(max_iter=1000)
        self.clf = OneVsRestClassifier(base_estimator)
        self.max_sx = 1

    def fit(self, X, s):
        """
        Fits the PGlin model to the data.

        Parameters
        ----------
        X : numpy.ndarray
            The data to fit the model to.
        s : numpy.ndarray
            The observed labels of the data.
        """
        self.clf.fit(X,s)
        sx = self.clf.predict_proba(X)[:,1]
        self.max_sx = np.max(sx)

    def predict(self, X):
        """
        Predicts the labels of the data.

        Parameters
        ----------
        X : numpy.ndarray
            The data to predict the labels of."""
        return np.where(self.predict_proba(X)[:, 1]>0.5,1,0) 

    def predict_proba(self, Xtest):
        """
        Predicts the probabilities of the data.

        Parameters
        ----------
        Xtest : numpy.ndarray
            The data to predict the probabilities of.
        """
        sx_test = self.clf.predict_proba(Xtest)[:, 1]
        ex_test = np.sqrt(self.max_sx*sx_test)
        yx_test = (1/ex_test) * sx_test
        yx_test[np.where(yx_test>1)]=1

        return np.array(list(zip(1 - yx_test, yx_test))) 


class PUGerychDeep(nn.Module):
    def __init__(self, clf, dims, device=0):
        """
        Initializes the deep PGlin model.

        Parameters
        ----------
        clf : str
            The type of classifier to use. Can be 'lr', 'mlp', 'cnn', or 'resnet'.
        dims : int
            The dimensionality of the data.
        device : int
            The device to run the model on.
        """
        super().__init__()
        self.device = "mps" if getattr(torch, 'has_mps', False) else "cuda:{}".format(device) if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        if clf == 'lr' or clf == 'mlp':
            assert dims != None, 'Classifier type {} requires specifying the dimensionality of the data.'.format(clf)

        if clf == 'lr':
            self.clf = LR(dims=dims).to(self.device)
        elif clf == 'mlp':
            self.clf = MLPReLU(dims=dims).to(self.device)
        elif clf == 'cnn':
            self.clf = FullCNN().to(self.device)
        elif clf == 'resnet':
            self.clf = Resnet().to(self.device)

        self.max_sx = 0

    def predict_proba(self, x):
        """
        Predicts the probabilities of the data.

        Parameters
        ----------
        x : torch.Tensor
            The data to predict the probabilities of.
        """
        with torch.no_grad():
            sx = self.clf(x, probabilistic=True)
            ex = torch.sqrt(self.max_sx * sx)
            yx = (1/ex) * sx
            return torch.clip(yx, max=1)

    def fit(self, trainloader, valloader, epochs=100, lr=1e-3):
        """
        Fits the deep PGlin model to the data.

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
        optimizer = torch.optim.Adam(self.clf.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()

        es = EarlyStopping()

        done = False
        for epoch in range(epochs):
            steps = list(enumerate(trainloader))
            pbar = tqdm.tqdm(steps)
            for i, data in pbar:

                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                optimizer.zero_grad()

                outputs = self.clf(inputs, probabilistic=False)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                loss.backward()
                optimizer.step()

                loss = loss.item()
                if i == len(steps) - 1:
                    self.eval()
                    v_loss = 0
                    with torch.no_grad():
                        for j, val_data in enumerate(valloader):
                            inputs, labels = val_data[0].to(self.device), val_data[1].to(self.device)
                            pred_y = self.clf(inputs, probabilistic=False)
                            v_loss += criterion(pred_y, labels.unsqueeze(1).float()).item()
                    v_loss = v_loss/(j + 1)
                    if es(self.clf, v_loss):
                        done = True
                    pbar.set_description(f"Epoch: {epoch}, tloss: {loss}, vloss: {v_loss:>7f}, EStop:[{es.status}]")
                    self.train()
                else:
                    pbar.set_description(f"Epoch: {epoch}, tloss {loss:}")
            if done == True:
                break

        self.eval()
        with torch.no_grad():
            for i, data in enumerate(trainloader):
                inputs = data[0].to(self.device)
                s = self.clf(inputs, probabilistic=True)
                self.max_sx = max(self.max_sx, max(s))
        self.train()
