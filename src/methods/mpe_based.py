import time
import numpy as np
from .algorithms import KM, TICE, DEDPUL, AlphaMax


METHOD_REGISTRY = {"km": KM, "tice": TICE, "dedpul": DEDPUL, "alphamax": AlphaMax}


class MPEBased:
    def __init__(
        self,
        mpe_method: str,
        mpe_args: dict,
        run_type: str,
        seed: int,
    ) -> None:
        self.mpe_method_name = mpe_method
        self.mpe_class = METHOD_REGISTRY[mpe_method]
        self.mpe_args = mpe_args
        self.run_type = run_type
        self.seed = seed

    def fit_estimate(self, X_data, y_data, s_data, X_ctrl=None):
        tstart = time.perf_counter()
        model = self.mpe_class(**self.mpe_args)
        est = model.estimate(X_data, s_data)
        internal_pi = 1 - est["alpha"]
        if self.run_type == "correction":
            # CONTROL FEATURES
            model_ctrl = self.mpe_class(**self.mpe_args)
            est_ctrl = model_ctrl.estimate(X_ctrl, s_data)
            internal_pi_ctrl = 1 - est_ctrl["alpha"]

            # COMBINED FEATURES
            X_comb = np.hstack([X_ctrl, X_data])
            model_comb = self.mpe_class(**self.mpe_args)
            est_comb = model_comb.estimate(X_comb, s_data)
            internal_pi_comb = 1 - est_comb["alpha"]

        run_time = time.perf_counter() - tstart
        results = {
            "method": "mpe",
            "model": self.mpe_method_name,
            "p_hat_test": internal_pi,
            "time": run_time,
        }
        if self.run_type == "correction":
            results.update(
                {
                    "p_hat_ctrl": internal_pi_ctrl,
                    "p_hat_comb": internal_pi_comb,
                    "p_hat_2MIA-comb": 2 * internal_pi - internal_pi_comb,
                    "p_hat_MIA-ctrl": internal_pi - internal_pi_ctrl,
                }
            )
        return results
