import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


class PGlin(BaseEstimator):
    def __init__(self):
        base_estimator = LogisticRegression(max_iter=1000)
        self.clf = OneVsRestClassifier(base_estimator)
        self.max_sx = 1

    def fit(self, X, s):
        """
        Parameters
        ----------
        X : numpy.ndarray
            The data to fit the model to.
        s : numpy.ndarray
            The observed labels of the data.
        """
        self.clf.fit(X, s)
        sx = self.clf.predict_proba(X)[:, 1]
        self.max_sx = np.max(sx)

    def predict(self, X):
        """
        Parameters
        ----------
        X : numpy.ndarray
            The data to predict the labels of."""
        return np.where(self.predict_proba(X)[:, 1] > 0.5, 1, 0)

    def predict_proba(self, X):
        """
        Predicts the probabilities of the data.

        Parameters
        ----------
        Xtest : numpy.ndarray
            The data to predict the probabilities of.
        """
        sx_test = self.clf.predict_proba(X)[:, 1]
        ex_test = np.sqrt(self.max_sx * sx_test)
        yx_test = (1 / ex_test) * sx_test
        yx_test[np.where(yx_test > 1)] = 1
        return np.column_stack((1 - yx_test, yx_test))
