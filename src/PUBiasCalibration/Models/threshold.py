import torch
import tqdm
import numpy as np

from torch import nn
from ..helper_files.classifiers import LR, MLPReLU, FullCNN, Resnet
from ..threshold_optimizer import ThresholdOptimizer
from ..helper_files.utils import EarlyStopping, sigmoid
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

def seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

class PUthreshold(BaseEstimator):
    """
    The implementation of our NTC-tMI model.
    """
    def __init__(self):
        """
        Initializes the threshold model.
        """
        self.clf = LogisticRegression(max_iter=1000)
        self.topt = None
        
    

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
        
        self.clf.fit(X,s)
        sx = self.clf.predict_proba(X)[:,1]
        
        sx[np.where(sx==1)] = 0.999
        sx[np.where(sx==0)] = 0.001
        
        lin_pred = np.log(sx/(1-sx))

        w0 = np.where(s==0)[0]
        lin_pred0 = lin_pred[w0]

        to = ThresholdOptimizer(k=3, n=100)
        t_opt = to.find_threshold(lin_pred0)
        
        self.topt = t_opt
        return self

    def predict(self, X):
        """
        Predicts the labels of the data.
       
        Parameters
        ----------
        X : numpy.ndarray
            The data to predict the labels of.
        """
        return np.where(self.predict_proba(X)>0.5,1,0) 
        
    def predict_proba(self, Xtest):
        """
        Predicts the probabilities of the data.
        
        Parameters
        ----------
        Xtest : numpy.ndarray
            The data to predict the probabilities of.
        """
        sx_test = self.clf.predict_proba(Xtest)[:,1]
        sx_test[np.where(sx_test==0)]=0.01
        sx_test[np.where(sx_test==1)]=0.99
        z_test = np.log(sx_test/(1-sx_test))
        yx_test = sigmoid(z_test - self.topt)
        return np.array(list(zip(1 - yx_test, yx_test))) 

class PUthresholddeep(nn.Module):
    """
    The implementation of our NTC-tMI model using deep PyTorch models.
    """
    def __init__(self, clf, dims=None, device=0) -> None:
        """
        Initializes the threshold model.
        
        Parameters
        ----------
        clf : str
            The type of classifier to use. Options are 'lr', 'mlp', 'cnn', and 'resnet'.
        dims : tuple
            The dimensionality of the data.
        device : int
            The device to use for training.
        """
        super().__init__()
        self.device = "mps" if getattr(torch, 'has_mps', False) else "cuda:{}".format(device) if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        if clf == 'lr' or clf == 'mlp':
            assert dims != None, 'Classifier type {} requires specifying the dimensionality of the data.'.format(clf)

        if clf == 'lr':
            self.ntc = LR(dims=dims).to(self.device)
        elif clf == 'mlp':
            self.ntc = MLPReLU(dims=dims).to(self.device)
        elif clf == 'cnn':
            self.ntc = FullCNN().to(self.device)
        elif clf == 'resnet':
            self.ntc = Resnet().to(self.device)

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
            scores = self.ntc(x, probabilistic=False)
            thresholded_scores = scores - self.threshold
            return torch.sigmoid(thresholded_scores)

    def calculate_optimal_threshold(self, trainloader):
        """
        Calculates the optimal threshold for the data.
        
        Parameters
        ----------
        trainloader : torch.utils.data.DataLoader
            The data to calculate the optimal threshold for.
        """
        dataset_size = len(trainloader.dataset)
        z_unlabeled = torch.empty(dataset_size).to(self.device)
        j = 0

        with torch.no_grad():
            for data in trainloader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
    
                inputs_unlabeled = inputs[labels == 0]
                unlabeled_length = len(inputs_unlabeled)
                z_ = self.ntc(inputs_unlabeled, probabilistic=False).squeeze().detach()
                z_unlabeled[j:j + unlabeled_length] = z_
                j += unlabeled_length


        z_unlabeled = z_unlabeled[:j].cpu().numpy()

        to = ThresholdOptimizer(k=3, n=1000)
        self.threshold = torch.tensor(to.find_threshold(z_unlabeled)).to(self.device)


    def fit(self, trainloader, valloader, epochs, lr=1e-3):
        """
        Fits the threshold model to the data.
        
        Parameters
        ----------
        trainloader : torch.utils.data.DataLoader
            The data to fit the model to.
        valloader : torch.utils.data.DataLoader
            The data to validate the model on.
        epochs : int
            The number of epochs to train the model for.
        lr : float
            The learning rate to use for training.
        """
        optimizer = torch.optim.Adam(self.ntc.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        
        es = EarlyStopping()

        done = False
        for epoch in range(epochs):
            steps = list(enumerate(trainloader))
            pbar = tqdm.tqdm(steps)
            for i, data in pbar:

                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                optimizer.zero_grad()
                outputs = self.ntc(inputs, probabilistic=False)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                loss.backward()
                optimizer.step()

                loss = loss.item()
                if i == len(steps) - 1:
                    v_loss = 0

                    for j, val_data in enumerate(valloader):
                        inputs, labels = val_data[0].to(self.device), val_data[1].to(self.device)
                        pred_y = self.ntc(inputs, probabilistic=False)
                        v_loss += criterion(pred_y, labels.unsqueeze(1).float()).item()

                    v_loss = v_loss/(j + 1)

                    if es(self.ntc, v_loss):
                        done = True

                    pbar.set_description(f"Epoch: {epoch}, tloss: {loss}, vloss: {v_loss:>7f}, EStop:[{es.status}]")
                
                else:
                    pbar.set_description(f"Epoch: {epoch}, tloss: {loss:}")
            
            if done == True:
                break
        
        self.calculate_optimal_threshold(trainloader)
        