import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, beta
from sklearn.linear_model import LinearRegression  # residualization
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler

from PUBiasCalibration.Models.LBEWithPrior import (
    LBEWithPrior,
    seed,
    _lbe_nu_estimate_p_robust,
)


def load_features(npz_path):
    """Robust loader that grabs the first ndarray inside npz and cleans NaNs."""
    data = np.load(npz_path, allow_pickle=True)
    arrays = [v for v in data.values() if isinstance(v, np.ndarray)]
    if not arrays:
        raise ValueError(f"No arrays found in {npz_path}")
    arr = arrays[0]
    if arr.dtype == object:
        try:
            arr = np.stack(arr)
        except Exception:
            arr = np.array([np.array(a) for a in arr])
    return np.nan_to_num(arr, nan=0.0)

def safe_logit(p):
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def prepare_data(name, seed, p, ss_len=2000, run_type="real"):
    """
    Prepare data for the experiment.

    Parameters
    ----------
    name : str
        The name of the dataset.
    seed : int
        The random seed.
    p : float
        Target member prevalence in the *test unlabeled* set (0..1)

    Returns
    -------
    X_test  : (n_test, d) float32
    y_test  : (n_test,)  int   # true labels: members=0, nonmembers=1
    s_test  : (n_test,)  int   # observed NU labels for test (here: all 0 = unlabeled)
    X_ctrl_test : (n_test, d) float32  # for correction only
    """
    np.random.seed(seed)

    folder = Path("../data")

    # Define patterns for members and non-members
    # Pattern 1: contains _real and _mem
    mem_pattern1 = f"{name}_*real*_mem*.npz"
    # Pattern 2: contains _real and _nonmem
    nonmem_pattern1 = f"{name}_*real*_nonmem*.npz"
    # Pattern 3: contains _ae_mem
    mem_pattern2 = f"{name}_*ae_mem*.npz"
    # Pattern 4: contains _ae_nonmem
    nonmem_pattern2 = f"{name}_*ae_nonmem*.npz"
    # Pattern 5: contains _from-mem
    mem_pattern3 = f"{name}_*from-mem*.npz"
    # Pattern 6: contains _from-nonmem
    nonmem_pattern3 = f"{name}_*from-nonmem*.npz"

    control_pattern = "ControlSEnsemble"
    # Pattern 7: contains _ae_mem
    mem_pattern4 = f"{control_pattern}_*ae_mem*.npz"
    # Pattern 8: contains _ae_nonmem
    nonmem_pattern4 = f"{control_pattern}_*ae_nonmem*.npz"
    # Pattern 9: contains _from-mem
    mem_pattern5 = f"{control_pattern}_*from-mem*.npz"
    # Pattern 10: contains _from-nonmem
    nonmem_pattern5 = f"{control_pattern}_*from-nonmem*.npz"


    # Try to find files matching the patterns
    mem_matches1 = list(folder.glob(mem_pattern1))
    nonmem_matches1 = list(folder.glob(nonmem_pattern1))
    mem_matches2 = list(folder.glob(mem_pattern2))
    nonmem_matches2 = list(folder.glob(nonmem_pattern2))
    mem_matches3 = list(folder.glob(mem_pattern3))
    nonmem_matches3 = list(folder.glob(nonmem_pattern3))
    mem_matches4 = list(folder.glob(mem_pattern4))
    nonmem_matches4 = list(folder.glob(nonmem_pattern4))
    mem_matches5 = list(folder.glob(mem_pattern5))
    nonmem_matches5 = list(folder.glob(nonmem_pattern5))

    # Select patterns based on run_type
    if run_type == "real" or "mean_mia_score" or "tail":
        # For real: Use pattern1 & pattern2 (files containing _real_*mem and _real_*nonmem)
        mem_matches = mem_matches1
        nonmem_matches = nonmem_matches1
        print(f"Run type: {run_type} - Using patterns 1 & 2 (real)")
    elif run_type == "synth":
        # For synth: Use pattern1 & pattern2 and pattern5 & pattern6 (files containing _real_*mem, _real_*nonmem, _from-mem, and _from-nonmem)
        mem_matches = mem_matches1 + mem_matches3
        nonmem_matches = nonmem_matches1 + nonmem_matches3
        print(f"Run type: {run_type} - Using patterns 1, 2, 5 & 6 (real and from-mem)")
    elif run_type == "ae_synth" or run_type == "correction":
        # For ae_synth: Use pattern3 & pattern4 and pattern5 & pattern6 (files containing _ae_mem, _ae_nonmem, _from-mem, and _from-nonmem)
        mem_matches = mem_matches2 + mem_matches3
        nonmem_matches = nonmem_matches2 + nonmem_matches3
        print(f"Run type: {run_type} - Using patterns 3, 4, 5 & 6 (ae and from-mem)")
    else:
        raise FileNotFoundError(
            f"No matching files found for run_type {run_type} with member patterns: {mem_pattern1}, {mem_pattern2}, {mem_pattern3} "
            f"and non-member patterns: {nonmem_pattern1}, {nonmem_pattern2}, {nonmem_pattern3} in {folder}"
        )

    mem_file = mem_matches[0]
    nonmem_file = nonmem_matches[0]

    if len(mem_matches) > 1:
        mem_file_generated = mem_matches[1]
        nonmem_file_generated = nonmem_matches[1]
    elif run_type == "real" or "mean_mia_score" or "tail":
        mem_file_generated = nonmem_matches[0] #in real we mix only real nonmembers.
        nonmem_file_generated = nonmem_matches[0]
    else:
        print(f"No generated files found for run_type {run_type} with member patterns: {mem_pattern1}, {mem_pattern2}, {mem_pattern3} ")
        return 1
    print(f"Using files: {mem_file.name} and {nonmem_file.name}")


    members = np.load(
        mem_file,
        allow_pickle=True,
    )
    X_mem = members["data"]
    nonmembers = np.load(
        nonmem_file,
        allow_pickle=True,
    )
    X_nonmem = nonmembers["data"]

    members_generated = np.load(
        mem_file_generated,
        allow_pickle=True,
    )
    X_mem_generated = members_generated["data"]
    nonmembers_generated = np.load(
        nonmem_file_generated,
        allow_pickle=True,
    )
    X_nonmem_generated = nonmembers_generated["data"]

    if run_type =="correction":

        X_ctrl_ae_mem = load_features(mem_matches4[0])
        X_ctrl_ae_nonmem = load_features(nonmem_matches4[0])
        X_ctrl_mem_generated   = load_features(mem_matches5[0])
        X_ctrl_nonmem_generated   = load_features(nonmem_matches5[0])

    # shuffle with one permutation for all arrays
    # Generate a single permutation large enough for all arrays
    perm = np.random.permutation(max(len(X_mem), len(X_nonmem), len(X_mem_generated), len(X_nonmem_generated)))
    # Apply the same permutation to all arrays (using appropriate slices)
    X_mem = X_mem[perm[:len(X_mem)]]
    X_nonmem = X_nonmem[perm[:len(X_nonmem)]]
    X_mem_generated = X_mem_generated[perm[:len(X_mem_generated)]]
    X_nonmem_generated = X_nonmem_generated[perm[:len(X_nonmem_generated)]]

    if run_type == "correction":
        X_ctrl_ae_mem = X_ctrl_ae_mem[perm[:len(X_ctrl_ae_mem)]]
        X_ctrl_ae_nonmem = X_ctrl_ae_nonmem[perm[:len(X_ctrl_ae_nonmem)]]
        X_ctrl_mem_generated = X_ctrl_mem_generated[perm[:len(X_ctrl_mem_generated)]]
        X_ctrl_nonmem_generated = X_ctrl_nonmem_generated[perm[:len(X_ctrl_nonmem_generated)]]

    # -----------------------------
    # Test set construction (unlabeled only)
    # -----------------------------
    n_pos_test = int(ss_len * p)  # members inside suspect set
    n_unl_test_nonmem = ss_len - n_pos_test  # non-members inside test unlabeled

    if run_type == "real" or "mean_mia_score" or "tail":
        X_test_nonmem = np.concatenate(
            [
                X_mem_generated[:int(ss_len/2)],
                X_nonmem_generated[int(ss_len/2):ss_len],
            ]
        )
    else:
        # generated data from examples in the suspect set
        X_test_nonmem = np.concatenate(
            [
                X_mem_generated[:n_pos_test],
                X_nonmem_generated[ss_len:ss_len + n_unl_test_nonmem],
            ]
        )

    X_test = np.concatenate(
        [
            X_test_nonmem,
            X_mem[:n_pos_test],
            X_nonmem[ss_len:ss_len+n_unl_test_nonmem],
        ]
    )

    y_test = np.concatenate(
        [
            np.ones(ss_len, dtype=int), #NM_labeled (possibly generated in run_tymes synth)
            np.zeros(n_pos_test, dtype=int),  # Umembers
            np.ones(n_unl_test_nonmem, dtype=int),  # Unon-members
        ]
    )

    s_test = np.concatenate(
        [
            np.ones(ss_len, dtype=int), #NM_labeled
            np.zeros(n_pos_test, dtype=int),  # Umembers
            np.zeros(n_unl_test_nonmem, dtype=int),  # Unon-members
        ]
    )

    # scale
    X_test = X_test.squeeze(1)
    scaler = MinMaxScaler()
    X_test = scaler.fit_transform(X_test)

    if run_type == "correction":
        X_ctrl_test_nonmem = np.concatenate(
            [
                X_ctrl_mem_generated[:n_pos_test],
                X_ctrl_nonmem_generated[ss_len:ss_len + n_unl_test_nonmem],
            ]
        )

        X_ctrl_test = np.concatenate(
            [
                X_ctrl_test_nonmem,
                X_ctrl_ae_mem[:n_pos_test],
                X_ctrl_ae_nonmem[ss_len:ss_len + n_unl_test_nonmem],
            ]
        )
        # scale
        X_ctrl_test = X_ctrl_test.squeeze(1)
        ols = LinearRegression().fit(X_ctrl_test, X_test)
        Xhat_test = ols.predict(X_ctrl_test) # take only ctrl features explaining MIA
        scaler = MinMaxScaler()
        X_ctrl_test = scaler.fit_transform(Xhat_test)

    return X_test, y_test, s_test, (X_ctrl_test if run_type == "correction" else None)


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


def experiment_lbe_with_prior(name, nsym, lbe_model, results_dir="../results", p=0.5, bins=10, device=1,
                              run_type="real", ss_len=2000):
    """
    Run the experiment with LBE using internal prior.

    Parameters
    ----------
    name : str
        The name of the dataset.
    nsym : int
        The number of iterations.
    p : float
        The probability value.
    results_dir : str
        The directory to save the results.
    bins : int
        The number of bins for the LBE model (default: 10).
    device : int
        Device to use for LBE training (default: 1).
    """

    metrics = [
        "acc",
        "bacc",
        "p_hat_test",
        "H0_p_value",
    ]
    if run_type == "correction":
        metrics += ["p_hat_ctrl", "p_hat_comb", 'p_hat_2MIA-comb', 'p_hat_MIA-ctrl']

    print("\n Method: LBE with internal prior")
    records = []
    for sym in np.arange(0, nsym, 1):
        X_test, y_test, s_test, X_ctrl_test = prepare_data(
            name=name, seed=sym, p=p, run_type=run_type, ss_len=ss_len,
        )
        np.random.seed(sym)
        seed(sym)
        if run_type == "mean_mia_score":
            sums = X_test.sum(axis=1)  # sum of features per example → shape (100,)
            print(X_test.shape, len(sums))
            print(X_test[0])
            # Min-max scale to [0, 1]
            scaled_sums = (sums - sums.min()) / (sums.max() - sums.min())
            pi = np.sum([1 if ss > 0.5 else 0 for ss in scaled_sums]) / len(sums)
            results = {
                "method": "mean_of_sum_of_scores",
                "acc": None,
                "bacc": None,
                "p_hat_test": pi,
            }
        else:
            # Use the LBEWithPrior model
            start_time = time.time()
            model = LBEWithPrior(kind=lbe_model, bins=bins, device=device)
            model.fit(X_test, s_test)
            end_time = time.time()
            run_time = end_time - start_time

            internal_pi = model.get_prior()

            if run_type == "correction" and X_ctrl_test is not None:
                X_comb_test = np.hstack([X_ctrl_test, X_test])

                # Fit LBEWithPrior on CONTROL-only and COMBINED
                start_time = time.time()
                mdl_ctrl = LBEWithPrior(kind=lbe_model, bins=bins, device=device)
                mdl_ctrl.fit(X_ctrl_test, s_test)

                mdl_comb = LBEWithPrior(kind=lbe_model, bins=bins, device=device)
                mdl_comb.fit(X_comb_test, s_test)
                end_time = time.time()
                run_time = end_time - start_time

                mdl_comb._fit_runtime = run_time  # stash for logging

                internal_pi_ctrl = mdl_ctrl.get_prior()
                internal_pi_comb = mdl_comb.get_prior()

            prob_y_test = model.predict_proba(X_test)[:, 1]

            def pick_tau_by_target_fpr(scores_neg, alpha):
                # tau is the (1-alpha) quantile so that FPR ~= alpha on negatives
                q = 1.0 - alpha
                return float(np.quantile(scores_neg, q, method="linear"))

            def no_positives_test(scores_S, scores_N, *, alpha_grid=(1e-3, 2e-3, 5e-3, 1e-2),
                                  combine="bonferroni", delta=0.05):
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

                    per_tau.append({
                        "alpha": a_eff, "tau": tau, "k": k, "rS": rS,
                        "pval": float(pval), "p_upper": float(p_upper)
                    })

                # Multiple-threshold control
                pvals = np.array([d["pval"] for d in per_tau])
                if combine == "bonferroni":
                    p_global = float(np.minimum(1.0, pvals.min() * len(pvals)))
                    pick = int(pvals.argmin())
                else:  # 'min' without correction (useful for exploration)
                    p_global = float(pvals.min());
                    pick = int(pvals.argmin())

                best = per_tau[pick]
                decision = (p_global < delta)

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

            p_value = (no_positives_test(prob_y_test[s_test == 0], prob_y_test[s_test == 1], delta=0.01))["p_value"]

            if run_type == "tail":
                def prior_from_tail_ratio(scores_S, scores_N, top_frac=0.2):
                    sS = np.sort(scores_S)
                    sN = np.sort(scores_N)
                    # candidate thresholds in the high tail of N
                    start = int((1 - top_frac) * len(sN))
                    taus = sN[start:]  # use quantiles from N's tail

                    # survival functions at taus
                    def surv(x, t):  # P(X >= t)
                        # fraction >= t via binary search
                        j = np.searchsorted(x, t, side='left')
                        return (len(x) - j) / len(x)

                    ratios = []
                    eps = 1.0 / max(1, len(sN))  # ridge to avoid 0
                    for t in taus:
                        rN = max(eps, surv(sN, t))
                        rS = surv(sS, t)
                        ratios.append(rS / rN)
                    kappa_hat = max(0.0, min(ratios))  # inf over tail
                    p_hat = max(0.0, min(1.0, 1.0 - kappa_hat))
                    return p_hat
                print(internal_pi)
                internal_pi = prior_from_tail_ratio(prob_y_test[s_test == 0], prob_y_test[s_test == 1])

                # internal_pi = estimate_p_roc_inversion(X_test.mean(axis=1)[s_test == 0], X_test.mean(axis=1)[s_test == 1], scores_U=X_test.mean(axis=1)[s_test == 0], pi=internal_pi)["p_hat"]
            # Flip labels
            prob_y_test = 1 - prob_y_test
            y_test = 1 - y_test

            acc = accuracy_score(y_test, np.where(prob_y_test > 0.5, 1, 0))
            bacc = balanced_accuracy_score(y_test, np.where(prob_y_test > 0.5, 1, 0))

            results = {
                "method": "lbe_with_internal_prior",
                "acc": acc,
                "bacc": bacc,
                "time": run_time,
                "p_hat_test": internal_pi,
                "H0_p_value": p_value,
            }

        if run_type == "correction" and X_ctrl_test is not None:
            results.update(
                {
                    "p_hat_ctrl": internal_pi_ctrl,
                    "p_hat_comb": internal_pi_comb,
                    'p_hat_2MIA-comb': 2*internal_pi-internal_pi_comb,
                    'p_hat_MIA-ctrl':internal_pi - internal_pi_ctrl,
                    "H0_p_value": p_value,
                }
            )

        print(results)
        records.append(results)

    df = pd.DataFrame(records)

    agg = df.groupby("method")[metrics].agg(["min", "mean", "std", "max"])

    # Format mean ± std
    rows = []
    for method in agg.index:
        row = {"method": method}
        for metric in metrics:
            row[f"{metric}_mean_std"] = (
                f"{agg[(metric, 'mean')][method]:.3f}±{agg[(metric, 'std')][method]:.3f}"
            )
        rows.append(row)

    formatted = pd.DataFrame(rows)

    # Create results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    model_name = name.split("/")[1] if "/" in name else name
    formatted.to_csv(
        f"{results_dir}/results_lbe_prior_{model_name}_{run_type}_len{ss_len}_bins{bins}_lbe{lbe_model}_p={p}.csv",
        index=False,
        sep="\t",
    )
    df.to_csv(
        f"{results_dir}/results_lbe_prior_full_{model_name}_{run_type}_len{ss_len}_bins{bins}_lbe{lbe_model}_p={p}.csv",
        index=False,
        sep="\t",
    )

    # Print the true unlabeled members ratio vs. estimated ratio
    print(f"\nTrue p: {p:.4f}")
    if run_type == "correction":
        print(
            f"Mean estimated ratio (p_hat_comb): {df['p_hat_comb'].mean():.4f} ± {df['p_hat_comb'].std():.4f}"
        )
        print(
            f"Mean estimated ratio (p_hat_ctrl): {df['p_hat_ctrl'].mean():.4f} ± {df['p_hat_ctrl'].std():.4f}"
        )
        print(
            f"Mean estimated ratio (p_hat_2MIA-comb): {df['p_hat_2MIA-comb'].mean():.4f} ± {df['p_hat_2MIA-comb'].std():.4f}"
        )
        print(
            f"Mean estimated ratio (p_hat_MIA-ctrl): {df['p_hat_MIA-ctrl'].mean():.4f} ± {df['p_hat_MIA-ctrl'].std():.4f}"
        )
    print(
        f"Mean estimated ratio (p_hat_test): {df['p_hat_test'].mean():.4f} ± {df['p_hat_test'].std():.4f}"
    )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-nsym", type=int, required=True, help="Number of iterations/runs"
    )
    parser.add_argument(
        "-prob", type=float, required=True, help="Probability value (between 0 and 1)"
    )
    parser.add_argument(
        "-lbe_model",
        type=str,
        default="LR",
        required=False,
        choices=["LR", "MLP"],
        help="LBE backbone model to use: LR (Logistic Regression, default) or MLP (Multi-layer Perceptron)",
    )
    parser.add_argument(
        "-data",
        type=str,
        required=True,
        help="Model to use. E.g.: var_24",
    )
    parser.add_argument(
        "-results",
        type=str,
        default="../results",
        required=False,
        help="Directory to save results (default: ../results)",
    )
    parser.add_argument(
        "-bins",
        type=int,
        default=10,
        required=False,
        help="Number of bins for the LBE model (default: 10)",
    )
    parser.add_argument(
        "-ss_len",
        type=int,
        default=2000,
        required=False,
        help="Number samples in the suspect set (default: 2000)",
    )
    parser.add_argument(
        "-device",
        type=int,
        default=1,
        required=False,
        help="Device to use for LBE training (default: 1)",
    )
    parser.add_argument(
        "-run_type",
        type=str,
        default="real",
        required=False,
        choices=["real", "synth", "ae_synth", "correction", "mean_mia_score", "tail"],
        help="Type of run to perform: real (pattern1&2), synth (pattern1&2 and pattern5&6), ae_synth (pattern3&4 and pattern5&6) (default: real)",
    )
    args = parser.parse_args()
    print(args.run_type)
    experiment_lbe_with_prior(
        args.data,
        args.nsym,
        args.lbe_model,
        args.results,
        ss_len = args.ss_len,
        p=args.prob,
        bins=args.bins,
        device=args.device,
        run_type=args.run_type,
    )


if __name__ == "__main__":
    main()
