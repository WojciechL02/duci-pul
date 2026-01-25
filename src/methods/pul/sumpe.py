import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from typing import Dict
from km import KM
from tice import TICE


class BestBin:
    def __init__(self):
        self.k_neighbors = 5

    def estimate(self, X, y):
        X_mixture = X[np.where(y == 0)[0], :]
        X_component = X[np.where(y == 1)[0], :]
        nbrs_U = NearestNeighbors(n_neighbors=self.k_neighbors).fit(X_mixture)
        dists_U, _ = nbrs_U.kneighbors(X_mixture)
        avg_dist_U = dists_U[:, -1]

        nbrs_P = NearestNeighbors(n_neighbors=self.k_neighbors).fit(X_component)
        dists_P, _ = nbrs_P.kneighbors(X_component)

        if X_component.shape[1] > 1:
            mean_P = np.mean(X_component, axis=0)
            mean_U = np.mean(X_mixture, axis=0)
            diff = mean_P - mean_U
            if np.linalg.norm(diff) < 1e-9:
                return 0.5
            w = diff / np.linalg.norm(diff)
            P_proj = X_component @ w
            U_proj = X_mixture @ w
        else:
            P_proj = X_component.flatten()
            U_proj = X_mixture.flatten()

        bins = np.histogram_bin_edges(np.concatenate([P_proj, U_proj]), bins="auto")
        hist_P, _ = np.histogram(P_proj, bins=bins, density=True)
        hist_U, _ = np.histogram(U_proj, bins=bins, density=True)

        valid_mask = hist_P > 1e-5
        if not np.any(valid_mask):
            return 0.0

        ratios = hist_U[valid_mask] / hist_P[valid_mask]
        alpha_est = np.min(ratios)
        return {"alpha": np.clip(alpha_est, 0.0, 1.0)}


METHOD_REGISTRY = {
    "best_bin": BestBin,
    "km": KM,
    "tice": TICE,
}


class SuMPE:
    """
    Subsampling Mixture Proportion Estimation (SuMPE).
    Uses subsampling and linear extrapolation to correct bias in MPE.
    """

    def __init__(
        self,
        base_estimator="best_bin",
        base_estimator_kwargs=None,
        ratios=None,
        n_repeats=3,
    ):
        """
        Args:
            base_estimator: An instance of an MPE estimator (must have .estimate(X, y)).
            ratios: List of subsampling ratios (default: [0.1, ..., 1.0]).
            n_repeats: Number of repetitions per ratio.
        """
        self.base_estimator = METHOD_REGISTRY[base_estimator]
        self.base_estimator_kwargs = (
            base_estimator_kwargs if base_estimator_kwargs is not None else {}
        )
        self.ratios = ratios if ratios is not None else [0.2, 0.4, 0.6, 0.8, 1.0]
        self.n_repeats = n_repeats

    def estimate(self, X, y) -> Dict[str, float]:
        """
        Estimates alpha using the SuMPE meta-algorithm.
        Args:
            X: Feature matrix.
            y: Labels (1 = Positive, 0 = Unlabeled).
        """
        X_P = X[y == 1]
        X_U = X[y == 0]
        n_U = len(X_U)

        if n_U < 50 or len(X_P) < 10:
            return {"alpha": 0.0}

        x_reg = []  # Ratios
        y_reg = []  # Estimates

        for r in self.ratios:
            ratio_estimates = []
            n_sub = int(n_U * r)

            if n_sub < 10:
                continue

            for _ in range(self.n_repeats):
                indices = np.random.choice(n_U, n_sub, replace=False)
                X_U_sub = X_U[indices]
                X_temp = np.vstack([X_P, X_U_sub])
                y_temp = np.concatenate([np.ones(len(X_P)), np.zeros(len(X_U_sub))])
                val = self._run_base_estimator(X_temp, y_temp)
                ratio_estimates.append(val)

            if ratio_estimates:
                x_reg.append(r)
                y_reg.append(np.mean(ratio_estimates))

        if len(x_reg) < 2:
            return {"alpha": np.mean(y_reg) if y_reg else 0.0}

        X_arr = np.array(x_reg).reshape(-1, 1)
        y_arr = np.array(y_reg)

        reg = LinearRegression().fit(X_arr, y_arr)
        estimated_value = reg.intercept_
        return {"alpha": float(np.clip(estimated_value, 0.0, 1.0))}

    def _run_base_estimator(self, X, y) -> float:
        estimator = self.base_estimator(**self.base_estimator_kwargs)
        results = estimator.estimate(X, y)
        return results["alpha"]
