
import torch
import tqdm
import torch.nn.functional as F
import numpy as np

from torch import nn
from copy import deepcopy
from ..helper_files.classifiers import MLPReLU, FullCNN, LR, Resnet
from ..helper_files.utils import EarlyStopping
from sklearn.base import BaseEstimator
from..helper_files.lbe.LBE import lbe_train, lbe_predict_proba

def seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

class LBE(BaseEstimator):
    def __init__(self) -> None:
        """
        Initializes the LBE model.
        """
        self.model = None
    
    def fit(self, X, s):
        """
        Fits the LBE model to the data.
        
        Parameters
        ----------
        X : numpy.ndarray
            The data to fit the model to.
        s : numpy.ndarray
            The observed labels of the data.
        """
        self.model = lbe_train(X, s, kind='LR', epochs=250)
    
    def predict_proba(self, X):
        """
        Predicts the probabilities of the data.
        
        Parameters
        ----------
        X : numpy.ndarray
            The data to predict the probabilities of.
        """
        y_pred = lbe_predict_proba(self.model, X)
        return np.array(list(zip(1 - y_pred, y_pred))) 

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
        self.device = "mps" if getattr(torch, 'has_mps', False) else "cuda:{}".format(device) if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        if clf == 'lr' or clf == 'mlp':
            assert dims != None, 'Classifier type {} requires specifying the dimensionality of the data.'.format(clf)
        
        if clf == 'lr':
            self.h = LR(dims=dims).to(self.device)
            self.eta = LR(dims=dims).to(self.device)
        elif clf == 'mlp':
            self.h = MLPReLU(dims=dims).to(self.device)
            self.eta = MLPReLU(dims=dims).to(self.device)
        elif clf == 'cnn':
            self.h = FullCNN().to(self.device)
            self.eta = FullCNN().to(self.device)
        elif clf == 'resnet':
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

            P_y_hat = torch.cat([P_y_hat_0.reshape(-1, 1), P_y_hat_1.reshape(-1, 1)], axis = 1)
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
            P_y_hat[:, 1] * log_1_minus_eta + P_y_hat[:, 0] * log_1_minus_eta
        )

        loss2 = torch.where(
            s == 1,
            P_y_hat[:, 1] * log_h + P_y_hat[:, 0] * log_1_minus_h,
            P_y_hat[:, 1] * log_h + P_y_hat[:, 0] * log_1_minus_h
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
                            inputs, labels = val_data[0].to(self.device), val_data[1].to(self.device)
                            pred_y = self.h(inputs, probabilistic=False)
                            v_loss += criterion(pred_y, labels.unsqueeze(1).float()).item()
                    v_loss = v_loss/(j + 1)
                    if es(self.h, v_loss):
                        done = True
                    pbar.set_description(f"Pre-training - Epoch: {epoch}, tloss: {loss}, vloss: {v_loss:>7f}, EStop:[{es.status}]")
                    self.train()
                else:
                    pbar.set_description(f"Pre-training - Epoch: {epoch}, tloss {loss:}")
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
                            inputs, s = val_data[0].to(self.device), val_data[1].to(self.device)
                            P_y_hat = self.E_step(inputs, s)
                            v_loss = self.loss(inputs, s, P_y_hat)
                    v_loss = v_loss/(j + 1)
                    if es(self.h, v_loss):
                        done = True
                    pbar.set_description(f"EM - Epoch: {epoch}, tloss: {loss}, vloss: {v_loss:>7f}, EStop:[{es.status}]")
                    self.train()
                else:
                    pbar.set_description(f"EM - Epoch: {epoch}, tloss {loss:}")
            
            self.h_frozen = deepcopy(self.h)
            self.eta_frozen = deepcopy(self.eta)

            if done == True:
                break

            
            
