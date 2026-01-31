import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression


class SAREM(BaseEstimator):
    def __init__(self):
        self.model = None
        self.propensity_model = None
        self.eps = 1e-4
        self.ll_eps = 0.0001
        self.slope_eps = 0.0001
        self.convergence_window = 10
        self.max_its = 500
        self.refit_classifier = True

    def fit(self, X, s):
        """
        Parameters
        ----------
        X : numpy.ndarray
            The data to fit the model to.
        s : numpy.ndarray
            The observed labels of the data.
        """
        clf_attributes = np.ones(X.shape[1]).astype(bool)
        propensity_attributes = np.ones(X.shape[1]).astype(bool)
        clf_model = LogisticRegressionPU()
        propensity_model = LogisticRegressionPU()
        self.model = LimitedFeaturesModel(clf_model, clf_attributes)
        self.propensity_model = LimitedFeaturesModel(
            propensity_model, propensity_attributes
        )

        initialize_simple(X, s, self.model, self.propensity_model)

        expected_prior_y1 = self.model.predict_proba(X)
        expected_propensity = self.propensity_model.predict_proba(X)
        expected_prior_y1 = np.clip(expected_prior_y1, self.eps, 1 - self.eps)
        expected_propensity = np.clip(expected_propensity, self.eps, 1 - self.eps)
        expected_posterior_y1 = expectation_y(expected_prior_y1, expected_propensity, s)

        # loglikelihood
        ll = self.loglikelihood_probs(expected_prior_y1, expected_propensity, s)
        loglikelihoods = [ll]

        # propensity slope
        past_propensities = np.zeros([int(len(s) - sum(s)), self.convergence_window])
        propensity_slope = []
        max_ll_improvements = []

        for i in range(self.max_its):
            # maximization
            self.propensity_model.fit(X, s, sample_weight=expected_posterior_y1)

            classification_s = np.concatenate(
                [
                    np.ones_like(expected_posterior_y1),
                    np.zeros_like(expected_posterior_y1),
                ]
            )
            classification_weights = np.concatenate(
                [expected_posterior_y1, 1 - expected_posterior_y1]
            )
            self.model.fit(
                np.concatenate([X, X], axis=0),
                classification_s,
                sample_weight=classification_weights,
            )

            # expectation
            expected_prior_y1 = self.model.predict_proba(X)
            expected_propensity = self.propensity_model.predict_proba(X)
            expected_prior_y1 = np.clip(expected_prior_y1, self.eps, 1 - self.eps)
            expected_propensity = np.clip(expected_propensity, self.eps, 1 - self.eps)
            expected_posterior_y1 = expectation_y(
                expected_prior_y1, expected_propensity, s
            )

            # loglikelihood
            ll = self.loglikelihood_probs(expected_prior_y1, expected_propensity, s)
            loglikelihoods.append(ll)

            # convergence
            push(past_propensities, expected_propensity[s == 0])
            if i > self.convergence_window:
                max_ll_improvement = (
                    max(loglikelihoods[-self.convergence_window :])
                    - loglikelihoods[-self.convergence_window]
                )
                max_ll_improvements.append(max_ll_improvement)
                average_abs_slope = np.average(np.abs(slope(past_propensities, axis=1)))
                propensity_slope.append(average_abs_slope)
                if (
                    average_abs_slope < self.slope_eps
                    and max_ll_improvement < self.ll_eps
                ):
                    break

        if self.refit_classifier:
            self.model.fit(X, s, e=expected_propensity)

    def predict_proba(self, X):
        """
        Parameters
        ----------
        X : numpy.ndarray
            The data to predict the probability of.
        """
        y_pred = self.model.predict_proba(X)
        return np.column_stack((1 - y_pred, y_pred))

    def loglikelihood_probs(self, class_probabilities, propensity_scores, labels):
        prob_labeled = class_probabilities * propensity_scores
        prob_unlabeled_pos = class_probabilities * (1 - propensity_scores)
        prob_unlabeled_pos = np.clip(prob_unlabeled_pos, self.eps, 1 - self.eps)
        prob_unlabeled_neg = 1 - class_probabilities
        prob_unlabeled_neg = np.clip(prob_unlabeled_neg, self.eps, 1 - self.eps)
        prob_pos_given_unl = prob_unlabeled_pos / (
            prob_unlabeled_pos + prob_unlabeled_neg
        )
        prob_neg_given_unl = 1 - prob_pos_given_unl

        return (
            labels * np.log(prob_labeled)
            + (1 - labels)
            * (
                prob_pos_given_unl * np.log(prob_unlabeled_pos)
                + prob_neg_given_unl * np.log(prob_unlabeled_neg)
            )
        ).mean()


def initialize_simple(instances, labels, classification_model, propensity_model):
    """Initialization with unlabeled=negative, but reweighting the examples so that the expected class prior is 0.5"""
    proportion_labeled = labels.sum() / labels.size
    classification_weights = (
        labels * (1 - proportion_labeled) + (1 - labels) * proportion_labeled
    )
    classification_model.fit(instances, labels, sample_weight=classification_weights)
    classification_expectation = classification_model.predict_proba(instances)
    propensity_model.fit(
        instances,
        labels,
        sample_weight=(labels + (1 - labels) * classification_expectation),
    )


def expectation_y(expectation_f, expectation_e, s):
    result = s + (1 - s) * (expectation_f * (1 - expectation_e)) / (
        1 - expectation_f * expectation_e
    )
    return result


def slope(array, axis=0):
    """Calculate the slope of the values in ar over dimension "axis". The values are assumed to be equidistant."""
    if axis == 1:
        array = array.transpose()

    n = array.shape[0]
    norm_x = np.asarray(range(n)) - (n - 1) / 2
    auto_cor_x = np.square(norm_x).mean(0)
    avg_y = array.mean(axis=0)
    norm_y = array - avg_y
    cov_x_y = np.matmul(norm_y.transpose(), norm_x) / n
    result = cov_x_y / auto_cor_x
    if axis == 1:
        result = result.transpose()
    return result


def push(array_queue, new_array):
    array_queue[:, :-1] = array_queue[:, 1:]
    array_queue[:, -1] = new_array


class LogisticRegressionPU(LogisticRegression):
    def __init__(self):
        super(LogisticRegressionPU, self).__init__(l1_ratio=0, solver="liblinear")

    def fit(self, x, s, e=None, sample_weight=None):
        if e is None:
            super().fit(x, s, sample_weight)
        else:
            Xp, Yp, Wp = self._make_propensity_weighted_data(x, s, e, sample_weight)
            super().fit(Xp, Yp, Wp)

    def _make_propensity_weighted_data(self, x, s, e, sample_weight=None):
        weights_pos = s / e
        weights_neg = (1 - s) + s * (1 - 1 / e)
        if sample_weight is not None:
            weights_pos = sample_weight * weights_pos
            weights_neg = sample_weight * weights_neg

        Xp = np.concatenate([x, x])
        Yp = np.concatenate([np.ones_like(s), np.zeros_like(s)])
        Wp = np.concatenate([weights_pos, weights_neg])
        return Xp, Yp, Wp


class NoFeaturesModel:
    def __init__(self, prior=0.5):
        self.prior = prior

    def fit(self, x, y, sample_weight=None):
        if sample_weight is None:
            self.prior = y.mean()
        else:
            self.prior = (y * sample_weight).mean()

    def predict_proba(self, x):
        return np.ones(x.shape[0]) * self.prior


class LimitedFeaturesModel:
    def __init__(self, model, features):
        self.model = model
        self.features = features

    def predict_proba(self, x):
        pr = self.model.predict_proba(x[:, self.features])
        if np.ndim(pr) > 1 and np.shape(pr)[1] > 1:
            pr = pr[:, 1]
        return pr

    def fit(self, x, y, e=None, sample_weight=None):
        if isinstance(self.model, LogisticRegressionPU):
            self.model.fit(x[:, self.features], y, e, sample_weight)
        else:
            self.model.fit(x[:, self.features], y, sample_weight)
        return self
