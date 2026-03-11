import time
import numpy as np
from typing import Dict


BASELINES = ["mia_guess", "mia_score"]


def mia_guess():
    pass


def mia_score():
    pass


METHOD_REGISTRY = {
    "mia_guess": mia_guess,
    "mia_score": mia_score, 
}


class Baseline:
    def __init__(
        self,
        method: str,
        run_type: str,
        seed: int,
    ) -> None:
        if method not in METHOD_REGISTRY.keys():
            raise ValueError(f"No known PUL method {method}")
        self.method_name = method
        self.alg = METHOD_REGISTRY[method]
        self.run_type = run_type
        self.seed = seed

    def fit_estimate(self, X_data, y_data, s_data, X_ctrl=None) -> Dict:
        start_time = time.perf_counter()
        # TODO: MEAN MIA
        pi_estimated = self.alg()  # TODO: fill with arguments
        run_time = time.perf_counter() - start_time

        results = {
            "method": "pul",
            "model": self.method_name,
            "p_hat_test": pi_estimated,
            "time": run_time,
        }
        return results
