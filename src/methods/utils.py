import os
import numpy as np
from scipy.stats import binom, beta
import matplotlib.pyplot as plt


def pick_tau_by_target_fpr(scores_neg, alpha):
    # tau is the (1-alpha) quantile so that FPR ~= alpha on negatives
    q = 1.0 - alpha
    return float(np.quantile(scores_neg, q, method="linear"))


def no_positives_test(
    scores_S,
    scores_N,
    *,
    alpha_grid=(1e-3, 2e-3, 5e-3, 1e-2),
    combine="bonferroni",
    delta=0.05,
):
    """
    One-sided H0: p=0 test + conservative upper bound on p (no TPR needed).
    - Scans several target FPRs (alpha_grid), guards small-sample noise.
    - combine: 'bonferroni' or 'min' for multiple-threshold control.
    - delta: confidence level for the (1-delta) upper bound on p.
    Returns:
      dict with p_value (global), decision, best_tau, alpha_at_tau, k_obs,
      p_upper (conservative (1-delta) upper confidence bound on mixture proportion p).
    """
    scores_S = np.asarray(scores_S, float)
    scores_N = np.asarray(scores_N, float)
    nS = scores_S.size
    if nS == 0:
        raise ValueError("scores_S is empty.")
    if scores_N.size == 0:
        raise ValueError("scores_N is empty.")

    per_tau = []
    for a in alpha_grid:
        # skip alphas too small for the size of N (avoid zero/unstable tails)
        min_fpr = 1.0 / max(10, scores_N.size)
        a_eff = max(a, min_fpr)
        tau = pick_tau_by_target_fpr(scores_N, a_eff)
        k = int(np.sum(scores_S >= tau))
        rS = k / nS

        # p-value for H0: K ~ Bin(nS, a_eff), test is Pr(K >= k)
        pval = binom.sf(k - 1, nS, a_eff)

        # Conservative (1-delta) upper bound on rS via Clopper–Pearson,
        # then translate to upper bound on p with worst-case TPR=1:
        # p <= (r_upper - alpha)/(1 - alpha), clipped to [0,1].
        # CP upper bound on r given k successes in nS trials:
        r_upper = beta.ppf(1 - delta, k + 1, nS - k) if k < nS else 1.0
        p_upper = max(0.0, min(1.0, (r_upper - a_eff) / (1.0 - a_eff)))

        per_tau.append(
            {
                "alpha": a_eff,
                "tau": tau,
                "k": k,
                "rS": rS,
                "pval": float(pval),
                "p_upper": float(p_upper),
            }
        )

    # Multiple-threshold control
    pvals = np.array([d["pval"] for d in per_tau])
    if combine == "bonferroni":
        p_global = float(np.minimum(1.0, pvals.min() * len(pvals)))
        pick = int(pvals.argmin())
    else:  # 'min' without correction (useful for exploration)
        p_global = float(pvals.min())
        pick = int(pvals.argmin())

    best = per_tau[pick]
    decision = p_global < delta

    return {
        "p_value": p_global,
        "decision_reject_H0_p_equals_0": decision,
        "best_tau": best["tau"],
        "alpha_at_tau": best["alpha"],
        "k_obs": best["k"],
        "rS": best["rS"],
        "p_upper": best["p_upper"],
        "details_per_tau": per_tau,
    }


def plot_hist_with_excess(hU, hN, edges, real_p, estimated_p):
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
    ax.set_title(f"Estimated p={estimated_p:.2f}")

    histograms_dir = "./histograms"
    output_png = os.path.join(histograms_dir, f"p={real_p}.png")
    output_pdf = os.path.join(histograms_dir, f"p={real_p}.pdf")
    save_dir = os.path.dirname(output_png)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def nu_estimate_p_robust(
    scores_U: np.ndarray,
    scores_N: np.ndarray,
    *,
    real_p: float = None,
    bins: int = 80,
    tail_trim: float = 0.001,
    min_bin_count_N: int = 5,
    min_bin_count_U: int = 1,
    relax: float | str = "auto",
    auto_relax_scale: float = 2.0,
    show_histogram: bool = False,
    debug: bool = False,
) -> float:
    if debug:
        diagnose_scores(scores_U, scores_N)
        diagnose_ratio_curve(scores_U, scores_N)
        diagnose_heavy_tail(scores_U, scores_N)
        diagnose_bootstrap_bias(scores_U, scores_N)
        diagnose_clumping(scores_U)

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

    cU, _ = np.histogram(U, bins=edges)
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
    # piN_hat = float(np.clip(np.min(ratios), 0.0, 1.0))
    piN_hat = np.percentile(ratios, 10.0)

    estimated_p = float(np.clip(1.0 - piN_hat, 0.0, 1.0))
    if show_histogram:
        plot_hist_with_excess(hU_, piN_hat * hN_, edges, real_p, estimated_p)
    return estimated_p


def diagnose_clumping(scores_U):
    u_vals = np.unique(scores_U)
    print(f"--- DIAGNOSIS 3 ---")
    print(f"Total Samples: {len(scores_U)}")
    print(f"Unique Scores: {len(u_vals)}")
    if len(u_vals) < 100:
        print("WARNING: Severe Clumping detected. Add noise before estimation.")
    else:
        print("Score resolution looks fine.")


def diagnose_bootstrap_bias(scores_U, scores_N, n_boot=50):
    estimates = []

    # Run 50 iterations of resampling
    for _ in range(n_boot):
        # Sample with replacement
        U_resample = np.random.choice(scores_U, size=len(scores_U), replace=True)
        N_resample = np.random.choice(scores_N, size=len(scores_N), replace=True)

        # Use your current best estimator function
        est = nu_estimate_p_robust(U_resample, N_resample)
        estimates.append(est)

    mean_est = np.mean(estimates)
    std_est = np.std(estimates)

    print(f"--- DIAGNOSIS 2 ---")
    print(f"Mean Estimate: {mean_est:.3f}")
    print(f"Std Dev: {std_est:.3f}")
    print(f"95% CI: [{mean_est - 2 * std_est:.3f}, {mean_est + 2 * std_est:.3f}]")


def diagnose_heavy_tail(scores_U, scores_N):
    plt.figure(figsize=(10, 5))

    # Use Log Scale for Y-axis to see the "tail" behavior
    plt.hist(scores_U, bins=50, alpha=0.5, density=True, label="Suspect (U)", log=True)
    plt.hist(
        scores_N, bins=50, alpha=0.5, density=True, label="Reference (N)", log=True
    )

    plt.title("Diagnosis 1: Log-Density Tail Check")
    plt.xlabel("Classifier Score")
    plt.ylabel("Log Density")
    plt.legend()
    plt.show()


def diagnose_scores(U, N):
    print(f"--- DIAGNOSIS ---")
    print(f"Range U: [{np.min(U):.4f}, {np.max(U):.4f}]")
    print(f"Range N: [{np.min(N):.4f}, {np.max(N):.4f}]")

    # Check Skewness (Median vs Mean)
    print(f"N Mean: {np.mean(N):.4f}, N Median: {np.median(N):.4f}")

    # Check Linear Binning Failure
    counts, _ = np.histogram(N, bins=50)
    empty_bins = np.sum(counts == 0)
    print(f"With 50 Linear Bins, {empty_bins} bins are totally empty.")

    # Check if mass is concentrated
    mass_in_top_bin = counts[-1] / len(N)
    print(f"Mass in highest bin: {mass_in_top_bin * 100:.1f}%")


def diagnose_ratio_curve(scores_U, scores_N, bins=20):
    U = np.array(scores_U)
    N = np.array(scores_N)

    # Quantile Bins
    both = np.concatenate([U, N])
    edges = np.quantile(both, np.linspace(0, 1, bins + 1))

    cU, _ = np.histogram(U, bins=edges)
    cN, _ = np.histogram(N, bins=edges)

    # Avoid div by zero
    mask = cN >= 5
    prob_U = cU[mask] / len(U)
    prob_N = cN[mask] / len(N)

    ratios = prob_U / prob_N

    # PLOT
    plt.figure(figsize=(10, 4))
    plt.plot(ratios, marker="o", linestyle="-", label="U/N Ratio")
    plt.axhline(0.8, color="green", linestyle="--", label="True Min (0.8)")
    plt.axhline(
        np.min(ratios),
        color="red",
        linestyle="--",
        label=f"Detected Min ({np.min(ratios):.2f})",
    )
    plt.title("The 'Ratio Curve' Diagnosis")
    plt.xlabel("Bin Index (Low Score -> High Score)")
    plt.ylabel("Density Ratio")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
