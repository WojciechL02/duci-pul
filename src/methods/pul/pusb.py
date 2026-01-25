import torch
from torch import nn
import numpy as np
from scipy import optimize
from sklearn.base import BaseEstimator


class PUSB(BaseEstimator):
    def __init__(self, class_prior, X_test, y_test) -> None:
        """
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
        self.clf = PULinearKernel(pi=self.pi)
        self.X_test = X_test
        self.y_test = y_test

    def fit(self, X, s):
        """
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
        prob_y_test = self.clf.test_pred(
            self.x_test_kernel, self.pu_res, self.y_test, quant=True, pi=self.pi
        )
        return np.array(list(zip(1 - prob_y_test, prob_y_test)))


class PUSBLoss(nn.Module):
    def __init__(self, class_prior):
        """
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

        loss_p = (
            -self.pi
            * torch.sum(torch.nn.functional.logsigmoid(outputs[positives]))
            / (nb_p)
        )
        loss_n = -torch.sum(torch.nn.functional.logsigmoid(-outputs[unlabeled])) / (
            nb_u
        ) + self.pi * torch.sum(torch.nn.functional.logsigmoid(-outputs[positives])) / (
            nb_p
        )
        # For deep learning: Take the max between 0 and loss_n for loss_n in neural networks, because the second part of the negative loss is not capped in the negative direction. See Kato et al.
        return loss_p + torch.clamp(loss_n, min=0)


class PULinearKernel:
    def __init__(self, pi):
        self.pi = pi
        self.loss_func = lambda g: self.loss(g)

    def loss(self, g):
        g = np.logaddexp(0, -g)
        return g

    def pu(self, x, b, t, reg):
        xp = x[t == 1]
        xu = x[t == 0]
        n1 = len(xp)
        # if n1 == 0:
        # print(n1)
        n0 = len(xu)
        gp = np.dot(xp, b)
        gu = np.dot(xu, b)
        loss_u = self.loss_func(-gu)
        J1 = -(self.pi / n1) * np.sum(gp)
        J0 = (1 / n0) * np.sum(loss_u)
        J = J1 + J0 + reg * np.dot(b, b)
        return J

    def prob(self, x, b):
        x = self.x
        g = np.dot(x, b)
        prob = 1 / (1 + np.exp(-g))
        return prob

    def optimize(self, x, t, x_test):
        x_train, x_test, lda_chosen = self.kernel_cv(x, t, x_test)
        # print(np.sum(t))
        res = self.minimize(x_train, t, lda_chosen)
        return res, x_test

    def minimize(self, x, t, reg):
        b0 = np.zeros(x.shape[1])
        func = lambda b: self.pu(x, b, t, reg)
        grad = lambda b: self.gradient(x, b, t, reg)
        self.result = optimize.minimize(func, b0, jac=grad, method="BFGS")
        self.result = self.result.x
        return self.result

    def gradient(self, x, b, t, reg):
        xp = x[t == 1]
        xu = x[t == 0]
        n1 = len(xp)
        n0 = len(xu)
        g = np.dot(xu, b)
        z = 1 / (1 + np.exp(-g))
        dg = np.sum(xp, axis=0) / n1
        grad = -self.pi * dg + np.dot(z.T, xu) / n0 + reg * b
        return grad

    def test(self, x, b, t, quant=True, pi=False):
        theta = 0
        f = np.dot(x, b)
        if quant is True:
            temp = np.copy(f)
            temp = np.sort(temp)
            theta = temp[np.int32(np.floor(len(x) * (1 - pi)))]
        pred = np.zeros(len(x))
        pred[f > theta] = 1
        acc = np.mean(pred == t)
        return acc

    def test_pred(self, x, b, t, quant=True, pi=False):
        theta = 0
        f = np.dot(x, b)
        if quant is True:
            temp = np.copy(f)
            temp = np.sort(temp)
            theta = temp[np.int32(np.floor(len(x) * (1 - pi)))]
        pred = np.zeros(len(x))
        pred[f > theta] = 1
        return pred

    def dist(self, x, T=None, num_basis=False):
        (d, n) = x.shape
        if not num_basis:
            num_basis = 300

        idx = np.random.permutation(n)[0:num_basis]
        C = x[:, idx]

        XC_dist = distance_squared(x, C)
        TC_dist = distance_squared(T, C)
        CC_dist = distance_squared(C, C)
        return XC_dist, TC_dist, CC_dist, n, num_basis

    def kernel_cv(
        self,
        x_train,
        t,
        x_test,
        folds=5,
        num_basis=False,
        sigma_list=None,
        lda_list=None,
    ):
        x_train, x_test = x_train.T, x_test.T
        XC_dist, TC_dist, CC_dist, n, num_basis = self.dist(x_train, x_test, num_basis)
        # setup the cross validation
        cv_fold = np.arange(folds)  # normal range behaves strange with == sign
        cv_split0 = np.floor(np.arange(n) * folds / n)
        cv_index = cv_split0[np.random.permutation(n)]
        # set the sigma list and lambda list
        if sigma_list == None:
            sigma_list = np.array([0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20])
        if lda_list == None:
            lda_list = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0])

        score_cv = np.zeros((len(sigma_list), len(lda_list)))

        for sigma_idx, sigma in enumerate(sigma_list):

            # pre-sum to speed up calculation
            h_cv = []
            t_cv = []
            for k in cv_fold:
                h_cv.append(np.exp(-XC_dist[:, cv_index == k] / (2 * sigma**2)))
                t_cv.append(t[cv_index == k])

            for k in range(folds):
                # print(h0_cv[0])
                # calculate the h vectors for training and test
                count = 0
                for j in range(folds):
                    if j == k:
                        hte = h_cv[j].T
                        tte = t_cv[j]
                    else:
                        if count == 0:
                            htr = h_cv[j].T
                            ttr = t_cv[j]
                            count += 1
                        else:
                            htr = np.append(htr, h_cv[j].T, axis=0)
                            ttr = np.append(ttr, t_cv[j], axis=0)

                one = np.ones((len(htr), 1))
                htr = np.concatenate([htr, one], axis=1)
                one = np.ones((len(hte), 1))
                hte = np.concatenate([hte, one], axis=1)
                for lda_idx, lda in enumerate(lda_list):
                    res = self.minimize(htr, ttr, lda)
                    # calculate the solution and cross-validation value

                    score = self.pu(hte, res, tte, lda)
                    """
                    if math.isnan(score):
                        code.interact(local=dict(globals(), **locals()))
                    """

                    score_cv[sigma_idx, lda_idx] = score_cv[sigma_idx, lda_idx] + score

        # get the minimum
        (sigma_idx_chosen, lda_idx_chosen) = np.unravel_index(
            np.argmin(score_cv), score_cv.shape
        )
        sigma_chosen = sigma_list[sigma_idx_chosen]
        lda_chosen = lda_list[lda_idx_chosen]

        x_train = np.exp(-XC_dist / (2 * sigma_chosen**2)).T
        x_test = np.exp(-TC_dist / (2 * sigma_chosen**2)).T

        one = np.ones((len(x_train), 1))
        x_train = np.concatenate([x_train, one], axis=1)
        one = np.ones((len(x_test), 1))
        x_test = np.concatenate([x_test, one], axis=1)

        return x_train, x_test, lda_chosen


def distance_squared(X, C):
    """
    Calculates the squared distance between X and C.
    XC_dist2 = CalcDistSquared(X, C)
    [XC_dist2]_{ij} = ||X[:, j] - C[:, i]||2
    :param X: dxn: First set of vectors
    :param C: d:nc Second set of vectors
    :return: XC_dist2: The squared distance nc x n
    """
    Xsum = np.sum(X**2, axis=0).T
    Csum = np.sum(C**2, axis=0)
    XC_dist = Xsum[np.newaxis, :] + Csum[:, np.newaxis] - 2 * np.dot(C.T, X)
    return XC_dist
