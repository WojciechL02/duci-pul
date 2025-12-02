import matplotlib.pyplot as plt
import scienceplots
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


def plot_hist_with_excess(hU, hN, edges, model_name, real_p, estimated_p):
    plt.style.use(["science", "no-latex"])
    bw = np.diff(edges)
    bw[bw == 0] = 1e-12

    fig, ax = plt.subplots(figsize=(7, 4))
    fontsize = 16
    fontweight = "heavy"

    # Plot the two density curves
    ax.step(
        edges,
        np.append(hU, hU[-1]),
        where="post",
        color="orange",
        linewidth=1.5,
        label="Scaled unlabeled",
    )
    ax.step(
        edges,
        np.append(hN, hN[-1]),
        where="post",
        color="blue",
        linewidth=1.5,
        label="Scaled non-members",
    )
    # Plot the excess mass area
    ax.bar(
        x=edges[:-1],
        height=hU - hN,
        bottom=hN,
        width=bw,
        align="edge",
        linewidth=0,
        color="orange",
        alpha=0.25,
        zorder=-1,
        label="Excess mass = members (LBE)",
    )

    # Labels and formatting
    ax.set_xlabel("Score", fontsize=fontsize, fontweight=fontweight)
    ax.set_ylabel("Density", fontsize=fontsize, fontweight=fontweight)
    ax.set_xscale("log")
    ax.set_xlim(right=1.0)
    ax.tick_params(axis="both", labelsize=fontsize, width=1)
    ax.legend(
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        prop={"weight": fontweight, "size": 14},
    )
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_title(f"Real p={real_p:.2f} | Estimated p={estimated_p:.2f}")

    histograms_dir = "../histograms"
    output_png = os.path.join(histograms_dir, f"{model_name}_p={real_p}.png")
    output_pdf = os.path.join(histograms_dir, f"{model_name}_p={real_p}.pdf")
    save_dir = os.path.dirname(output_png)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def _lbe_nu_estimate_p_robust(
    scores_U: np.ndarray,
    scores_N: np.ndarray,
    *,
    model_name: str = None,
    real_p: float = None,
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

    edges = np.quantile(both, np.linspace(lo, hi, bins + 1))
    # edges = np.linspace(lo, hi, bins + 1)

    cU, _ = np.histogram(Uc, bins=edges)
    cN, _ = np.histogram(Nc, bins=edges)
    mask = (cN >= min_bin_count_N) & (cU >= min_bin_count_U)
    if not np.any(mask):
        mask = cN >= min_bin_count_N
    bw = np.diff(edges)
    bw[bw == 0] = 1e-12
    hU = cU / (nU * bw)
    hN = cN / (nN * bw)
    hU_ = cU / nU
    hN_ = cN / nN

    if relax == "auto":
        relax = auto_relax_scale / min(nU, nN)
    denom = np.maximum(hN[mask], 1e-12)
    ratios = (hU[mask] + float(relax)) / denom
    piN_hat = float(np.clip(np.min(ratios), 0.0, 1.0))

    estimated_p = float(np.clip(1.0 - piN_hat, 0.0, 1.0))
    if model_name is not None:
        plot_hist_with_excess(
            hU_, piN_hat * hN_, edges, model_name, real_p, estimated_p
        )
    return estimated_p


def estimate_p_test(
    lbe_model,
    X_test,
    s_test,
    X_train,
    s_train,
    *,
    proba_index: int = 1,
):
    """
    Estimate prevalence p on the test set using:
      - test unlabeled scores (s_test == 0)
    No retraining is done.

    Parameters
    ----------
    lbe_model : LBEWithPrior
        A fitted LBEWithPrior instance (must already contain trained model).
    X_test : ndarray
        Test data (contains only unlabeled).
    s_test : ndarray
        Observed labels for test (0=unlabeled, 1=known negatives).
    X_train : ndarray
        Training data used to extract N reference scores.
    s_train : ndarray
        Observed labels for train (0=unlabeled, 1=known negatives).
    proba_index : int
        Index of the "U/member-like" probability column in predict_proba (default 1).

    Returns
    -------
    float
        Estimated member prevalence p_hat on the test set.
    """
    if lbe_model.model is None:
        raise RuntimeError("LBE model must be fitted first (call fit() on train).")

    # predict_proba on both splits
    probs_test = lbe_model.predict_proba(X_test)
    probs_train = lbe_model.predict_proba(X_train)

    scores_test_U = probs_test[:, proba_index][s_test == 0]
    scores_train_N = probs_train[:, proba_index][s_train == 1]

    # orientation check (flip if needed)
    if np.mean(scores_test_U) < np.mean(scores_train_N):
        scores_test_U = 1.0 - scores_test_U
        scores_train_N = 1.0 - scores_train_N

    # run robust LBE on those two score sets
    p_hat_test = _lbe_nu_estimate_p_robust(
        scores_test_U,
        scores_train_N,
        bins=lbe_model.bins,
        tail_trim=lbe_model.tail_trim,
        min_bin_count_N=lbe_model.min_bin_count_N,
        min_bin_count_U=lbe_model.min_bin_count_U,
        relax=lbe_model.relax,
    )
    return p_hat_test


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

    def estimate_p_on(self, X_new, s_new):
        """
        Reuse the trained model to estimate prevalence p on a new split/dataset.
        s_new uses the same convention (0=U, 1=N). No retraining happens here.
        """
        if self.model is None:
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
