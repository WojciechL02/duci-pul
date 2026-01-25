from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


class PUbasic(BaseEstimator):

    def __init__(self):
        """
        Initializes a fully labeled model.
        """
        base_estimator = LogisticRegression(max_iter=1000)
        self.clf = OneVsRestClassifier(base_estimator)

    def fit(self, X, y):
        """
        Fits the fully labeled model to the data.

        Parameters
        ----------
        X : numpy.ndarray
            The data to fit the model to.
        y : numpy.ndarray
            The observed labels of the data.
        """
        self.clf.fit(X, y=y)
        return self

    def predict(self, X):
        """
        Predicts the labels of the data.

        Parameters
        ----------
        X : numpy.ndarray
            The data to predict the labels of.
        """
        return self.clf.predict(X)

    def predict_proba(self, X):
        """
        Predicts the probabilities of the data.

        Parameters
        ----------
        X : numpy.ndarray
            The data to predict the probabilities of.
        """
        return self.clf.predict_proba(X)
