import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import binom, beta


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def safe_logit(p):
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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


def save_results(records: list, metrics: list, results_dir: str, metadata):
    df = pd.DataFrame(records)
    agg = df.groupby("method")[metrics].agg(["min", "mean", "std", "max"])

    rows = []
    for method in agg.index:
        row = {"method": method}
        for metric in metrics:
            row[f"{metric}_mean_std"] = (
                f"{agg[(metric, 'mean')][method]:.3f}±{agg[(metric, 'std')][method]:.2f}"
            )
        rows.append(row)

    formatted = pd.DataFrame(rows)
    os.makedirs(results_dir, exist_ok=True)

    model_name = metadata.target
    if "mpe_args" in list(metadata.method.keys()):
        filename = f"{metadata.method.name}_{model_name}_{metadata.run_type}_len{metadata.ss_len}_p={metadata.prob}.csv"
    else:
        pul_args = "_"
        if len(metadata.method.pul_args) > 0:
            pul_args = (
                "_"
                + "_".join(f"{k}={v}" for k, v in metadata.method.pul_args.items())
                + "_"
            )
        filename = f"{metadata.method.name}_{model_name}_{metadata.run_type}_len{metadata.ss_len}_bins{metadata.method.method_args.bins}{pul_args}p={metadata.prob}.csv"

    formatted.to_csv(
        os.path.join(results_dir, filename),
        index=False,
        sep="\t",
    )
    df.to_csv(
        os.path.join(results_dir, filename),
        index=False,
        sep="\t",
    )
    return df


def print_summary(df, config):
    print("\nSUMMARY:")
    print(100 * "-")
    print(f"True p: {config.prob:.4f}")
    if config.run_type == "correction":
        print(
            f"p_hat_comb: {df['p_hat_comb'].mean():.3f} ± {df['p_hat_comb'].std():.2f}"
        )
        print(
            f"p_hat_ctrl: {df['p_hat_ctrl'].mean():.3f} ± {df['p_hat_ctrl'].std():.2f}"
        )
        print(
            f"p_hat_2MIA-comb: {df['p_hat_2MIA-comb'].mean():.3f} ± {df['p_hat_2MIA-comb'].std():.2f}"
        )
        print(
            f"p_hat_MIA-ctrl: {df['p_hat_MIA-ctrl'].mean():.3f} ± {df['p_hat_MIA-ctrl'].std():.2f}"
        )
    print(f"p_hat_test: {df['p_hat_test'].mean():.3f} ± {df['p_hat_test'].std():.2f}")
    print(100 * "-")
