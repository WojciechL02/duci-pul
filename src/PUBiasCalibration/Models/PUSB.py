import torch
import tqdm
import numpy as np

from torch import nn

from ..helper_files.classifiers import FullCNN, MLPReLU, LR, Resnet
from ..helper_files.utils import EarlyStopping
from sklearn.base import BaseEstimator
from ..helper_files.pusb import pusb_linear_kernel as pusb
from ..helper_files.pusb.pusb_linear_kernel import PU

def seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    pusb.seed(seed)


class PUSB(BaseEstimator):
    def __init__(self, class_prior, X_test, y_test) -> None:
        """
        Initializes the PUSB model.
        
        Parameters
        ----------
        class_prior : float
            The class prior of the data.
        X_test : numpy.ndarray
            The test data.
        y_test : numpy.ndarray
            The test labels.
        """
        self.pi = class_prior
        self.clf = PU(pi=self.pi)
        self.X_test = X_test
        self.y_test = y_test

    def fit(self, X, s):
        """
        Fits the PUSB model to the data.
        
        Parameters
        ----------
        X : numpy.ndarray
            The data to fit the model to.
        s : numpy.ndarray
            The observed labels of the data.
        """
        self.pu_res, self.x_test_kernel = self.clf.optimize(X, s, self.X_test)
    
    def predict_proba(self, X):
        """
        Predicts the probabilities of the data.
        
        Parameters
        ----------
        X : numpy.ndarray
            The data to predict the probabilities of.
        """
        prob_y_test = self.clf.test_pred(self.x_test_kernel, self.pu_res, self.y_test, quant=True, pi=self.pi)
        return np.array(list(zip(1 - prob_y_test, prob_y_test)))


class PUSBLoss(nn.Module):
    # This loss function is not entirely deterministic, leading to slightly different results on different runs.
    def __init__(self, class_prior):
        """
        Initializes the PUSBLoss function.
        
        Parameters
        ----------
        class_prior : float
            The class prior of the data.
        """
        super().__init__()
        self.pi = class_prior
    
    def forward(self, outputs, targets):
        """
        Calculates the loss of the model.

        Parameters
        ----------
        outputs : torch.Tensor
            The outputs of the model.
        targets : torch.Tensor
            The targets of the model.
        """
        positives = targets == 1
        unlabeled = targets == 0
        nb_p = max(1, torch.sum(positives))
        nb_u = max(1, torch.sum(unlabeled))
        
        loss_p = -self.pi*torch.sum(torch.nn.functional.logsigmoid(outputs[positives]))/(nb_p)
        loss_n = -torch.sum(torch.nn.functional.logsigmoid(-outputs[unlabeled]))/(nb_u) + self.pi*torch.sum(torch.nn.functional.logsigmoid(-outputs[positives]))/(nb_p)

        # For deep learning: Take the max between 0 and loss_n for loss_n in neural networks, because the second part of the negative loss is not capped in the negative direction. See Kato et al.
        return loss_p + torch.clamp(loss_n, min=0)

class PUSBdeep(nn.Module):
   
    def __init__(self, clf, dims, class_prior, device=0) -> None:
        """
        Initializes the PUSBdeep model.
        
        Parameters
        ----------
        clf : str
            The classifier to use.
        dims : int
            The dimensionality of the data.
        class_prior : float
            The class prior of the data.
        device : int
            The device to use for training.
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

        self.class_prior = class_prior
        self.threshold = None

    def predict_proba(self, x):
        """
        Predicts the probabilities of the data.
        
        Parameters
        ----------
        x : torch.Tensor
            The data to predict the probabilities of.
        """
        assert self.threshold != None, 'The model has to be fit before predictions can be made.'
        with torch.no_grad():
            h = self.clf(x, probabilistic=False)
            return torch.sigmoid(h - self.threshold)
        
    def set_threshold(self, trainloader):
        """
        Sets the threshold for the model using the class prior.
        
        Parameters
        ----------
        trainloader : torch.utils.data.DataLoader
            The data to set the threshold for.
        """
        self.eval()
        Z = torch.tensor([], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            for data in trainloader:
                inputs = data[0].to(self.device)
                pred = self.clf(inputs, probabilistic=False)
                Z = torch.cat((Z, pred))
        index_threshold = (1 - self.class_prior) * len(Z)
        sorted_Z, _ = torch.sort(Z, dim=0)
        self.threshold = sorted_Z[int(index_threshold)]
        self.train()


    def fit(self, trainloader, valloader, epochs=100, lr=1e-3):
        """
        Fits the PUSBdeep model to the data.
        
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
        criterion = PUSBLoss(self.class_prior)
        
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

        self.set_threshold(trainloader)