import os
import time
import numpy as np
import torch
from sklearn.base import BaseEstimator
from typing import Dict
from .PUBiasCalibration.Models.LBE import LBE
from .PUBiasCalibration.Models.SAREM import SAREM
from .PUBiasCalibration.Models.threshold import PUthreshold
from .utils import nu_estimate_p_robust, no_positives_test


METHOD_REGISTRY = {
    "lbe": LBE,
    "sarem": SAREM,
    "threshold": PUthreshold,
}


class PULWithPrior(BaseEstimator):
    """
    NU setting:
      s==1 -> known negatives (N)
      s==0 -> unlabeled (U)
    Trains a scorer with lbe_train, uses predict_proba, and runs robust histogram LBE on that score.
    """

    def __init__(
        self,
        method,
        bins=30,
        relax="auto",
        tail_trim=0.001,
        min_bin_count_N=5,
        min_bin_count_U=1,
        proba_index=1,
        device=0,
    ):
        self.method = method
        self.bins = bins
        self.relax = relax
        self.tail_trim = tail_trim
        self.min_bin_count_N = min_bin_count_N
        self.min_bin_count_U = min_bin_count_U
        self.proba_index = proba_index
        self.device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.pi = None  # estimated prevalence on the last fit() dataset

    def fit(self, X, s):
        self.method.fit(X, s)
        self.pi = self._estimate_p_from_X_and_s(X, s)
        return self

    def predict_proba(self, X):
        probs = self.method.predict_proba(X)
        # probs = np.asarray(probs)
        # if probs.ndim == 1:
        #     probs = np.vstack([1 - probs, probs]).T
        return probs

    def prior_(self):
        if self.pi is None:
            raise RuntimeError("Call fit() first.")
        return self.pi

    def get_prior(self):
        return self.prior_()

    def estimate_p_on(self, X_new, s_new):
        """
        Reuse the trained model to estimate prevalence p on a new split/dataset.
        s_new uses the same convention (0=U, 1=N). No retraining happens here.
        """
        if self.method.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self._estimate_p_from_X_and_s(X_new, s_new)

    def _estimate_p_from_X_and_s(self, X, s):
        probs = self.predict_proba(X)  # (n,2)
        scores = probs[:, self.proba_index]  # prob of “U/member-like” class
        scores_U = scores[s == 0]
        scores_N = scores[s == 1]

        if np.mean(scores_U) < np.mean(scores_N):
            scores_U = 1.0 - scores_U
            scores_N = 1.0 - scores_N

        p_hat = nu_estimate_p_robust(
            scores_U,
            scores_N,
            bins=self.bins,
            tail_trim=self.tail_trim,
            min_bin_count_N=self.min_bin_count_N,
            min_bin_count_U=self.min_bin_count_U,
            relax=self.relax,
        )
        return p_hat

    def estimate_p_from_scores(self, scores: np.ndarray, s: np.ndarray):
        """
        If you already have predict_proba scores for a dataset (prob of U/member-like),
        pass them here with s (0=U, 1=N) to run LBE directly.
        """
        scores = np.asarray(scores).ravel()
        scores_U = scores[s == 0]
        scores_N = scores[s == 1]
        if np.mean(scores_U) < np.mean(scores_N):
            scores_U = 1.0 - scores_U
            scores_N = 1.0 - scores_N
        return nu_estimate_p_robust(
            scores_U,
            scores_N,
            bins=self.bins,
            tail_trim=self.tail_trim,
            min_bin_count_N=self.min_bin_count_N,
            min_bin_count_U=self.min_bin_count_U,
            relax=self.relax,
        )


class PULBased:
    def __init__(
        self,
        pul_method: str,
        pul_args: dict,
        method_args: dict,
        run_type: str,
        seed: int,
    ) -> None:
        if pul_method not in METHOD_REGISTRY.keys():
            raise ValueError(f"No known PUL method {pul_method}")
        self.pul_method_name = pul_method
        self.pul = METHOD_REGISTRY[pul_method]
        self.pul_args = pul_args
        self.method_args = method_args
        self.run_type = run_type
        self.seed = seed

    def fit_estimate(self, X_data, y_data, s_data, X_ctrl=None) -> Dict:
        start_time = time.perf_counter()
        model = PULWithPrior(self.pul(**self.pul_args), **self.method_args)
        model.fit(X_data, s_data)
        run_time = time.perf_counter() - start_time

        internal_pi = model.get_prior()

        if self.run_type == "correction":
            if X_ctrl is None:
                raise ValueError(
                    "X_ctrl is None while the 'correction' run type is chosen!"
                )

            X_comb = np.hstack([X_ctrl, X_data])

            start_time = time.perf_counter()
            model_ctrl = self.pul(self.pul_args)
            model_ctrl.fit(X_ctrl, s_data)

            model_comb = self.pul(self.pul_args)
            model_comb.fit(X_comb, s_data)
            run_time = time.perf_counter() - start_time

            model_comb._fit_runtime = run_time  # stash for logging

            internal_pi_ctrl = model_ctrl.get_prior()
            internal_pi_comb = model_comb.get_prior()

        prob_y_test = model.predict_proba(X_data)[:, 1]

        p_value = (
            no_positives_test(
                prob_y_test[s_data == 0], prob_y_test[s_data == 1], delta=0.01
            )
        )["p_value"]

        results = {
            "method": "pul",
            "model": self.pul_method_name,
            "p_hat_test": internal_pi,
            "H0_p_value": p_value,
            "time": run_time,
        }

        if self.run_type == "correction":
            results.update(
                {
                    "p_hat_ctrl": internal_pi_ctrl,
                    "p_hat_comb": internal_pi_comb,
                    "p_hat_2MIA-comb": 2 * internal_pi - internal_pi_comb,
                    "p_hat_MIA-ctrl": internal_pi - internal_pi_ctrl,
                    "H0_p_value": p_value,
                }
            )
        return results
