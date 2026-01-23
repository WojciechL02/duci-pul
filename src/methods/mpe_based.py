import time
import numpy as np
from .PUBiasCalibration.helper_files.km import KM


METHOD_REGISTRY = {
    "km": KM,
}


class MPEBased:
    def __init__(
        self,
        mpe_method: str,
        mpe_args: dict,
        run_type: str,
        seed: int,
    ) -> None:
        self.mpe_method_name = mpe_method
        self.mpe = METHOD_REGISTRY[mpe_method](**mpe_args)
        self.run_type = run_type
        self.seed = seed

    def fit_estimate(self, X_data, y_data, s_data, X_ctrl=None):
        tstart = time.perf_counter()
        X_mixture = np.unique(X_data[np.where(s_data == 0)[0], :], axis=0)
        X_component = np.unique(X_data[np.where(s_data == 1)[0], :], axis=0)
        km1, km2 = self.mpe.estimate(X_mixture, X_component)
        run_time = time.perf_counter() - tstart
        results = {
            "method": "mpe",
            "model": self.mpe_method_name,
            "p_hat_test": 1 - km1,
            "km2": 1 - km2,
            "time": run_time,
        }
        return results
