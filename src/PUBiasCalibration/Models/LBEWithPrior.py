import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from PUBiasCalibration.helper_files.lbe.LBE import lbe_train, lbe_predict_proba
from sklearn.base import BaseEstimator


def seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def plot_hist_with_excess(hU, hN, edges, estimated_p):
    centers = 0.5 * (edges[1:] + edges[:-1])
    bw = np.diff(edges)

    fig, ax = plt.subplots(figsize=(7, 4))

    # Plot the two density curves
    ax.step(
        edges[:-1],
        hU,
        where="post",
        color="orange",
        linewidth=2,
        label="Unlabeled mixture",
    )
    ax.step(
        edges[:-1],
        hN,
        where="post",
        color="blue",
        linewidth=2,
        label="Scaled non-members",
    )

    # Fill only where hU > hN (excess mass)
    ax.fill_between(
        centers,
        hU,
        hN,
        where=(hU > hN),
        color="orange",
        alpha=0.4,
        interpolate=True,
        label="Excess mass = members (LBE)",
    )

    # Labels and formatting
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_title(
        f"LBE estimation of p using histograms | Estimated p = {estimated_p:.2f}"
    )
    # Create histograms directory if it doesn't exist
    histograms_dir = "../histograms"
    if not os.path.exists(histograms_dir):
        os.makedirs(histograms_dir)

    fig.savefig(os.path.join(histograms_dir, "hist.png"), bbox_inches="tight")
    plt.close(fig)


def _lbe_nu_estimate_p_robust(
    scores_U: np.ndarray,
    scores_N: np.ndarray,
    *,
    bins: int = 80,
    tail_trim: float = 0.001,
    min_bin_count_N: int = 5,
    min_bin_count_U: int = 1,
    relax: float | str = "auto",
    auto_relax_scale: float = 2.0,
) -> float:
    U = np.asarray(scores_U, float).ravel()
    N = np.asarray(scores_N, float).ravel()
    nU, nN = len(U), len(N)
    if nU == 0 or nN == 0:
        raise ValueError("scores_U and scores_N must be non-empty.")
    both = np.concatenate([U, N])
    lo = np.quantile(both, tail_trim)
    hi = np.quantile(both, 1 - tail_trim)
    Uc = U[(U >= lo) & (U <= hi)]
    Nc = N[(N >= lo) & (N <= hi)]

    # NEW CODE:
    edges = np.quantile(both, np.linspace(lo, hi, bins + 1))
    # OLD CODE:
    # edges = np.linspace(lo, hi, bins + 1)

    cU, _ = np.histogram(Uc, bins=edges)
    cN, _ = np.histogram(Nc, bins=edges)
    mask = (cN >= min_bin_count_N) & (cU >= min_bin_count_U)
    if not np.any(mask):
        mask = cN >= min_bin_count_N
    bw = np.diff(edges)
    bw[bw == 0] = 1e-12  # prevent division by zero
    hU = cU / (nU * bw)
    hN = cN / (nN * bw)

    if relax == "auto":
        relax = auto_relax_scale / min(nU, nN)
    denom = np.maximum(hN[mask], 1e-12)
    ratios = (hU[mask] + float(relax)) / denom
    piN_hat = float(np.clip(np.min(ratios), 0.0, 1.0))

    estimated_p = float(np.clip(1.0 - piN_hat, 0.0, 1.0))
    # plot_hist_with_excess(hU, piN_hat * hN, edges, estimated_p)
    return estimated_p


class LBEWithPrior(BaseEstimator):
    """
    NU setting:
      s==1 -> known negatives (N)
      s==0 -> unlabeled (U)
    Trains a scorer with lbe_train, uses predict_proba, and runs robust histogram LBE on that score.
    """

    def __init__(
        self,
        kind="LR",
        bins=30,
        relax="auto",
        tail_trim=0.001,
        min_bin_count_N=5,
        min_bin_count_U=1,
        proba_index=1,
        device=1,
    ):
        self.kind = kind
        self.bins = bins
        self.relax = relax
        self.tail_trim = tail_trim
        self.min_bin_count_N = min_bin_count_N
        self.min_bin_count_U = min_bin_count_U
        self.proba_index = proba_index
        self.device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.pi = None  # estimated prevalence on the last fit() dataset

        if kind not in ["MLP", "LR"]:
            raise ValueError(
                f"Classifier kind '{kind}' not supported. Use 'MLP' or 'LR'."
            )

    def fit(self, X, s):
        # Train the scorer once (do NOT train on test when evaluating)
        self.model = lbe_train(X, s, kind=self.kind, epochs=250, device=self.device)
        # Estimate p on this dataset
        self.pi = self._estimate_p_from_X_and_s(X, s)
        return self

    def predict_proba(self, X):
        probs = lbe_predict_proba(self.model, X)  # <- your function
        probs = np.asarray(probs)
        if probs.ndim == 1:
            probs = np.vstack([1 - probs, probs]).T
        return probs

    def prior_(self):
        if self.pi is None:
            raise RuntimeError("Call fit() first.")
        return self.pi

    def get_prior(self):
        return self.prior_()

    # ---------- NEW: estimate p on ANY new dataset without retraining ----------
    def estimate_p_on(self, X_new, s_new):
        """
        Reuse the trained model to estimate prevalence p on a new split/dataset.
        s_new uses the same convention (0=U, 1=N). No retraining happens here.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self._estimate_p_from_X_and_s(X_new, s_new)

    # ---------- Helper: shared logic ----------
    def _estimate_p_from_X_and_s(self, X, s):
        probs = self.predict_proba(X)  # (n,2)
        scores = probs[:, self.proba_index]  # prob of “U/member-like” class
        scores_U = scores[s == 0]
        scores_N = scores[s == 1]

        # Orientation check for probabilities; flip if needed
        if np.mean(scores_U) < np.mean(scores_N):
            scores_U = 1.0 - scores_U
            scores_N = 1.0 - scores_N

        p_hat = _lbe_nu_estimate_p_robust(
            scores_U,
            scores_N,
            bins=self.bins,
            tail_trim=self.tail_trim,
            min_bin_count_N=self.min_bin_count_N,
            min_bin_count_U=self.min_bin_count_U,
            relax=self.relax,
        )
        return p_hat

    # ---------- Optional: estimate directly from scores without X ----------
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
        return _lbe_nu_estimate_p_robust(
            scores_U,
            scores_N,
            bins=self.bins,
            tail_trim=self.tail_trim,
            min_bin_count_N=self.min_bin_count_N,
            min_bin_count_U=self.min_bin_count_U,
            relax=self.relax,
        )
