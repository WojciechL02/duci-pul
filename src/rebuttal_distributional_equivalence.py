"""Rebuttal experiment: distributional equivalence between MIA(X_heldout) and MIA(X_synth).

Reviewer concern: synthetic "non-member" data may differ systematically from real
held-out (non-member) data, so using it as a proxy in the pipeline could bias the
PU-learning comparisons the paper relies on. We test that claim directly on the
MIA feature vectors (the columns are already per-example MIA-attack scores),
comparing X_heldout (real held-out non-members) against two candidate proxies:

  - X_synth : the raw synthetic non-member set (Synth condition).
  - X_ae    : the autoencoded suspect set (Synth + AE condition).

Two complementary, statistically independent pieces of evidence are produced:

  (A) MULTI-SEED DISTRIBUTIONAL COMPARISON (the main result). Over `--n_seeds`
      independent random subsamples we compute the RBF-kernel MMD^2 and the
      mean per-feature Wasserstein distance for heldout-vs-synth, heldout-vs-ae,
      and a within-class baseline (heldout vs heldout, disjoint halves). This
      turns each statistic into an empirical *distribution* (mean/std/95% CI)
      rather than a single seed-dependent number, and lets us run a proper
      two-sample test (Mann-Whitney U) between distributions instead of trusting
      one borderline permutation p-value:
        - synth  vs baseline : is the reviewers' concern real for raw synth?
        - ae     vs baseline : is there still a residual gap after the AE step?
        - ae     vs synth    : does the AE step significantly shrink the gap?

  (B) CONFIRMATORY SINGLE PERMUTATION TEST (secondary/classic check), using the
      largest practical single sample and many permutations, as in the original
      version of this script.
"""

import argparse

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import mannwhitneyu, wasserstein_distance

from data import load_data


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------
def median_heuristic_gamma(X: np.ndarray) -> float:
    """RBF bandwidth via the median heuristic: gamma = 1 / median(||x_i - x_j||^2)."""
    d2 = pdist(X, metric="sqeuclidean")
    d2 = d2[d2 > 0]
    med = np.median(d2) if len(d2) else 1.0
    return 1.0 / med if med > 0 else 1.0


def rbf_kernel_matrix(X: np.ndarray, gamma: float) -> np.ndarray:
    sq = np.sum(X * X, axis=1)
    sqdist = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(sqdist, 0.0, out=sqdist)
    return np.exp(-gamma * sqdist)


def mmd2_unbiased_from_kernel(K: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray) -> float:
    m, n = len(idx_a), len(idx_b)
    Kaa = K[np.ix_(idx_a, idx_a)]
    Kbb = K[np.ix_(idx_b, idx_b)]
    Kab = K[np.ix_(idx_a, idx_b)]
    term_aa = (Kaa.sum() - np.trace(Kaa)) / (m * (m - 1))
    term_bb = (Kbb.sum() - np.trace(Kbb)) / (n * (n - 1))
    term_ab = Kab.sum() / (m * n)
    return float(term_aa + term_bb - 2.0 * term_ab)


def mmd2_direct(A: np.ndarray, B: np.ndarray, gamma: float) -> float:
    """MMD^2 between A and B for a fixed, externally-supplied gamma (no permutation test)."""
    pooled = np.concatenate([A, B], axis=0)
    K = rbf_kernel_matrix(pooled, gamma)
    m = len(A)
    return mmd2_unbiased_from_kernel(K, np.arange(m), np.arange(m, m + len(B)))


def mmd_permutation_test(A, B, gamma, n_perm, rng):
    """MMD^2 between A and B + a classic permutation-test p-value (used for the confirmatory check)."""
    m, n = len(A), len(B)
    pooled = np.concatenate([A, B], axis=0)
    K = rbf_kernel_matrix(pooled, gamma)
    observed = mmd2_unbiased_from_kernel(K, np.arange(m), np.arange(m, m + n))

    all_idx = np.arange(m + n)
    perm_stats = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(all_idx)
        perm_stats[i] = mmd2_unbiased_from_kernel(K, perm[:m], perm[m:])

    p_value = (1 + np.sum(perm_stats >= observed)) / (n_perm + 1)
    return observed, p_value


def per_feature_wasserstein(A, B):
    return np.array([wasserstein_distance(A[:, j], B[:, j]) for j in range(A.shape[1])])


def subsample(X, n, rng):
    if n is None or len(X) <= n:
        return X
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx]


def mean_ci(x, alpha=0.05):
    lo, hi = np.percentile(x, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return x.mean(), x.std(), lo, hi


def common_language_effect_size(x, y):
    """P(a random draw from x > a random draw from y), via the Mann-Whitney U statistic."""
    U, _ = mannwhitneyu(x, y, alternative="two-sided")
    return U / (len(x) * len(y))


# ---------------------------------------------------------------------------
# Part A: multi-seed distributional comparison
# ---------------------------------------------------------------------------
def run_multiseed(X_heldout, X_synth, X_ae, gamma, args):
    n_seeds = args.n_seeds
    n_sub = args.mmd_samples
    n_features = X_heldout.shape[1]

    mmd_synth = np.empty(n_seeds)
    mmd_ae = np.empty(n_seeds)
    mmd_base = np.empty(n_seeds)
    wass_synth = np.empty((n_seeds, n_features))
    wass_ae = np.empty((n_seeds, n_features))
    wass_base = np.empty((n_seeds, n_features))

    for i in range(n_seeds):
        rng = np.random.default_rng(args.seed * 100_000 + i)

        a = subsample(X_heldout, n_sub, rng)
        b = subsample(X_synth, n_sub, rng)
        mmd_synth[i] = mmd2_direct(a, b, gamma)
        wass_synth[i] = per_feature_wasserstein(a, b)

        a2 = subsample(X_heldout, n_sub, rng)
        c = subsample(X_ae, n_sub, rng)
        mmd_ae[i] = mmd2_direct(a2, c, gamma)
        wass_ae[i] = per_feature_wasserstein(a2, c)

        perm = rng.permutation(len(X_heldout))
        half = len(perm) // 2
        H1, H2 = X_heldout[perm[:half]], X_heldout[perm[half:]]
        h1, h2 = subsample(H1, n_sub, rng), subsample(H2, n_sub, rng)
        mmd_base[i] = mmd2_direct(h1, h2, gamma)
        wass_base[i] = per_feature_wasserstein(h1, h2)

    return {
        "mmd": {"synth": mmd_synth, "ae": mmd_ae, "baseline": mmd_base},
        "wass": {"synth": wass_synth, "ae": wass_ae, "baseline": wass_base},
    }


def print_multiseed_summary(results, n_features, target):
    mmd = results["mmd"]
    wass_mean = {k: v.mean(axis=1) for k, v in results["wass"].items()}

    print("#" * 78)
    print(f"(A) MULTI-SEED DISTRIBUTIONAL COMPARISON  (n_seeds={len(mmd['synth'])})")
    print("#" * 78 + "\n")

    print(f"{'group':<28}{'metric':<10}{'mean':>10}{'std':>10}{'95% CI':>22}")
    for group in ["synth", "ae", "baseline"]:
        m, s, lo, hi = mean_ci(mmd[group])
        print(f"{'heldout vs ' + group:<28}{'MMD^2':<10}{m:>10.4g}{s:>10.4g}   [{lo:.4g}, {hi:.4g}]")
    for group in ["synth", "ae", "baseline"]:
        m, s, lo, hi = mean_ci(wass_mean[group])
        print(f"{'heldout vs ' + group:<28}{'Wass.':<10}{m:>10.4g}{s:>10.4g}   [{lo:.4g}, {hi:.4g}]")

    print("\nHypothesis tests (Mann-Whitney U, one-sided, over the per-seed MMD^2 / mean-Wasserstein values):\n")

    def report(name, x, y, alt, metric):
        stat, p = mannwhitneyu(x, y, alternative=alt)
        cles = common_language_effect_size(x, y)
        print(f"  [{metric}] {name:<38} p = {p:.4g}   (common-language effect size = {cles:.3f})")

    report("synth > baseline  (gap is real)", mmd["synth"], mmd["baseline"], "greater", "MMD^2")
    report("ae     > baseline  (residual gap)", mmd["ae"], mmd["baseline"], "greater", "MMD^2")
    report("ae     < synth     (AE helps)", mmd["ae"], mmd["synth"], "less", "MMD^2")
    report("synth > baseline  (gap is real)", wass_mean["synth"], wass_mean["baseline"], "greater", "Wass.")
    report("ae     > baseline  (residual gap)", wass_mean["ae"], wass_mean["baseline"], "greater", "Wass.")
    report("ae     < synth     (AE helps)", wass_mean["ae"], wass_mean["synth"], "less", "Wass.")

    # Per-feature aggregation across seeds
    rows = []
    for j in range(n_features):
        s_mean, s_std = results["wass"]["synth"][:, j].mean(), results["wass"]["synth"][:, j].std()
        a_mean, a_std = results["wass"]["ae"][:, j].mean(), results["wass"]["ae"][:, j].std()
        b_mean, b_std = results["wass"]["baseline"][:, j].mean(), results["wass"]["baseline"][:, j].std()
        rows.append(
            {
                "feature": j,
                "synth_mean": s_mean,
                "synth_std": s_std,
                "ae_mean": a_mean,
                "ae_std": a_std,
                "baseline_mean": b_mean,
                "baseline_std": b_std,
                "ratio_synth_to_baseline": s_mean / b_mean if b_mean else np.nan,
                "ratio_ae_to_baseline": a_mean / b_mean if b_mean else np.nan,
            }
        )
    df = pd.DataFrame(rows)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    print("\nPer-feature Wasserstein distance, mean ± std over seeds (with ratio to baseline):\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    seeds_df = pd.DataFrame(
        {
            "seed": np.arange(len(mmd["synth"])),
            "mmd_synth": mmd["synth"],
            "mmd_ae": mmd["ae"],
            "mmd_baseline": mmd["baseline"],
            "wass_mean_synth": wass_mean["synth"],
            "wass_mean_ae": wass_mean["ae"],
            "wass_mean_baseline": wass_mean["baseline"],
        }
    )
    seeds_df.to_csv(f"rebuttal_dist_equivalence_seeds_{target}.csv", index=False)
    df.to_csv(f"rebuttal_dist_equivalence_perfeature_{target}.csv", index=False)
    print(f"\nSaved per-seed values to rebuttal_dist_equivalence_seeds_{target}.csv")
    print(f"Saved per-feature summary to rebuttal_dist_equivalence_perfeature_{target}.csv")


# ---------------------------------------------------------------------------
# Part B: confirmatory single permutation test (classic check, large sample)
# ---------------------------------------------------------------------------
def run_confirmatory(X_heldout, X_synth, X_ae, gamma, args):
    print("\n" + "#" * 78)
    print("(B) CONFIRMATORY SINGLE PERMUTATION TEST")
    print("#" * 78 + "\n")

    rng = np.random.default_rng(args.seed)
    for label, X_other in [("synth", X_synth), ("ae", X_ae)]:
        a = subsample(X_heldout, args.confirm_samples, rng)
        b = subsample(X_other, args.confirm_samples, rng)
        mmd, p = mmd_permutation_test(a, b, gamma, args.confirm_perm, rng)
        print(
            f"heldout vs {label:<6}: MMD^2 = {mmd:.6g}   permutation p = {p:.4g}"
            f"  (n_perm={args.confirm_perm}, n={len(a)} vs {len(b)})"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Statistically robust, multi-seed distributional-equivalence test between "
        "MIA(X_heldout) and its two candidate proxies MIA(X_synth) / MIA(X_ae)."
    )
    parser.add_argument("--target", type=str, default="var_30")
    parser.add_argument("--data_dir", type=str, default="../data/standard")
    parser.add_argument("--n_seeds", type=int, default=30, help="Independent random subsamples for Part A.")
    parser.add_argument(
        "--mmd_samples",
        type=int,
        default=1500,
        help="Samples per group, per seed, for Part A (kernel matrix is O(n^2)).",
    )
    parser.add_argument("--confirm_samples", type=int, default=2000, help="Samples per group for Part B.")
    parser.add_argument("--confirm_perm", type=int, default=2000, help="Permutations for Part B.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed; all randomization derives from this.")
    args = parser.parse_args()

    rng0 = np.random.default_rng(args.seed)

    _, X_heldout = load_data(args.target, "real", args.data_dir)
    _, X_synth = load_data(args.target, "from", args.data_dir)
    _, X_ae = load_data(args.target, "ae", args.data_dir)

    X_heldout = X_heldout.reshape(X_heldout.shape[0], -1).astype(np.float64)
    X_synth = X_synth.reshape(X_synth.shape[0], -1).astype(np.float64)
    X_ae = X_ae.reshape(X_ae.shape[0], -1).astype(np.float64)
    n_features = X_heldout.shape[1]

    print(
        f"target={args.target}  X_heldout={X_heldout.shape}  X_synth={X_synth.shape}  "
        f"X_ae={X_ae.shape}\n"
    )

    # One shared bandwidth for every MMD^2 computed in this script (Part A and Part B),
    # so all reported values are on the same scale and directly comparable.
    gamma_pool = np.concatenate(
        [
            subsample(X_heldout, args.mmd_samples, rng0),
            subsample(X_synth, args.mmd_samples, rng0),
            subsample(X_ae, args.mmd_samples, rng0),
        ]
    )
    gamma = median_heuristic_gamma(gamma_pool)
    print(f"RBF gamma (median heuristic, shared across all tests): {gamma:.6g}\n")

    results = run_multiseed(X_heldout, X_synth, X_ae, gamma, args)
    print_multiseed_summary(results, n_features, args.target)

    run_confirmatory(X_heldout, X_synth, X_ae, gamma, args)


if __name__ == "__main__":
    main()
