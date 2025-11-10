import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression  # residualization
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from PUBiasCalibration.Models.LBEWithPrior import (
    LBEWithPrior,
    seed as lbe_seed,
    _lbe_nu_estimate_p_robust,
)
from PUBiasCalibration.helper_files.pu_metrics import (
    estimate_p_and_debias,   # standard PU (pi_hat / lowerbound)
    choose_threshold_nu,     # threshold selection using internal prior
    debias_target,           # debias with TPR/FPR to get p_hat
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --------------------------- helpers ---------------------------

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

# --------------------------- data prep (synthetic tp/fp + CONTROL, aligned) ---------------------------

def prepare_data(name, seed, p, unl_mem_ratio=0.5):
    """
    Returns:
      X_mia_train, X_mia_test, y_train, y_test, s_train, X_ctrl_train, X_ctrl_test
      (s_test is implicitly all zeros = unlabeled)
    """
    np.random.seed(seed)

    file_type = "loss" if name.startswith("mar_") else "cfg"

    # MIA features
    val_tp   = np.load(f"../data/{name}_llm_mia_{file_type}_5k_tp_s0.5_imagenet_val.npz",   allow_pickle=True)
    val_fp   = np.load(f"../data/{name}_llm_mia_{file_type}_5k_fp_s0.5_imagenet_val.npz",   allow_pickle=True)
    train_tp = np.load(f"../data/{name}_llm_mia_{file_type}_5k_tp_s0.5_imagenet_train.npz", allow_pickle=True)
    train_fp = np.load(f"../data/{name}_llm_mia_{file_type}_5k_fp_s0.5_imagenet_train.npz", allow_pickle=True)

    X_val_tp, X_val_fp = val_tp["data"], val_fp["data"]
    X_train_tp, X_train_fp = train_tp["data"], train_fp["data"]

    # CONTROL features (aligned 1:1 with MIA via identical permutations)
    control_prefix = "ControlSEnsemble"
    X_ctrl_train_fp = load_features(f"../data/{control_prefix}_5k_fp_s0.5_imagenet_train.npz")
    X_ctrl_train_tp = load_features(f"../data/{control_prefix}_5k_tp_s0.5_imagenet_train.npz")
    X_ctrl_val_fp   = load_features(f"../data/{control_prefix}_5k_fp_s0.5_imagenet_val.npz")
    X_ctrl_val_tp   = load_features(f"../data/{control_prefix}_5k_tp_s0.5_imagenet_val.npz")

    # identical permutations per split
    perm_val_tp   = np.random.permutation(len(X_val_tp))
    perm_val_fp   = np.random.permutation(len(X_val_fp))
    perm_train_tp = np.random.permutation(len(X_train_tp))
    perm_train_fp = np.random.permutation(len(X_train_fp))

    X_val_tp   = X_val_tp[perm_val_tp]
    X_val_fp   = X_val_fp[perm_val_fp]
    X_train_tp = X_train_tp[perm_train_tp]
    X_train_fp = X_train_fp[perm_train_fp]

    X_ctrl_val_tp   = X_ctrl_val_tp[perm_val_tp]
    X_ctrl_val_fp   = X_ctrl_val_fp[perm_val_fp]
    X_ctrl_train_tp = X_ctrl_train_tp[perm_train_tp]
    X_ctrl_train_fp = X_ctrl_train_fp[perm_train_fp]

    # ----- Train set construction (known negatives + unlabeled) -----
    n_lab_from_tp = int(2000 * unl_mem_ratio)
    n_lab_from_fp = 2000 - n_lab_from_tp
    X_mia_train_nonmem  = np.concatenate([X_val_tp[:n_lab_from_tp], X_val_fp[:n_lab_from_fp]])
    X_ctrl_train_nonmem = np.concatenate([X_ctrl_val_tp[:n_lab_from_tp], X_ctrl_val_fp[:n_lab_from_fp]])
    y_train_nonmem = np.ones(len(X_mia_train_nonmem), dtype=int)   # true nonmembers=1
    s_train_nonmem = np.ones(len(X_mia_train_nonmem), dtype=int)   # observed known negatives

    n_unl_mem = int(2000 * unl_mem_ratio)
    n_unl_nonmem = 2000 - n_unl_mem
    X_mia_train_unl  = np.concatenate([X_train_tp[:n_unl_mem], X_train_fp[:n_unl_nonmem]])
    X_ctrl_train_unl = np.concatenate([X_ctrl_train_tp[:n_unl_mem], X_ctrl_train_fp[:n_unl_nonmem]])
    y_train_unl = np.concatenate([np.zeros(n_unl_mem, dtype=int), np.ones(n_unl_nonmem, dtype=int)])
    s_train_unl = np.zeros(len(X_mia_train_unl), dtype=int)

    X_mia_train  = np.concatenate([X_mia_train_nonmem,  X_mia_train_unl])
    X_ctrl_train = np.concatenate([X_ctrl_train_nonmem, X_ctrl_train_unl])
    y_train = np.concatenate([y_train_nonmem, y_train_unl])
    s_train = np.concatenate([s_train_nonmem, s_train_unl])

    # ----- Test set (unlabeled only; fixed 2000) -----
    n_test_unl = 2000
    n_pos_test = int(n_test_unl * p)
    n_unl_test_nonmem = n_test_unl - n_pos_test

    X_mia_test_unl = np.concatenate([
        X_train_tp[n_unl_mem: n_unl_mem + n_pos_test],
        X_train_fp[n_unl_nonmem: n_unl_nonmem + n_unl_test_nonmem]
    ])
    y_test_unl = np.concatenate([
        np.zeros(n_pos_test, dtype=int),   # members
        np.ones(n_unl_test_nonmem, dtype=int)  # non-members
    ])
    X_ctrl_test_unl = np.concatenate([
        X_ctrl_train_tp[n_unl_mem: n_unl_mem + n_pos_test],
        X_ctrl_train_fp[n_unl_nonmem: n_unl_nonmem + n_unl_test_nonmem]
    ])

    # scale per space (MinMax) then Standardize for LBE
    X_mia_train = X_mia_train.squeeze(1); X_mia_test_unl = X_mia_test_unl.squeeze(1)
    X_ctrl_train = X_ctrl_train.squeeze(1); X_ctrl_test_unl = X_ctrl_test_unl.squeeze(1)

    mia_mm = MinMaxScaler().fit(X_mia_train)
    ctrl_mm = MinMaxScaler().fit(X_ctrl_train)
    X_mia_train = mia_mm.transform(X_mia_train); X_mia_test_unl = mia_mm.transform(X_mia_test_unl)
    X_ctrl_train = ctrl_mm.transform(X_ctrl_train); X_ctrl_test_unl = ctrl_mm.transform(X_ctrl_test_unl)

    # Standardize separately for each feature space before LBE training
    sc_mia  = StandardScaler().fit(X_mia_train)
    sc_ctrl = StandardScaler().fit(X_ctrl_train)
    Xm_tr = sc_mia.transform(X_mia_train); Xm_te = sc_mia.transform(X_mia_test_unl)
    Xc_tr = sc_ctrl.transform(X_ctrl_train); Xc_te = sc_ctrl.transform(X_ctrl_test_unl)

    return Xm_tr, Xm_te, y_train, y_test_unl, s_train, Xc_tr, Xc_te

# --------------------------- p_hat_test helper (like Scripts 1/2) ---------------------------

def estimate_p_test_from_trainN(lbe_model, X_test, s_test, X_train, s_train, *, proba_index: int = 1):
    """
    Estimate prevalence on test using test-U scores and train-N as reference.
    Uses the provided LBEWithPrior model's predict_proba scores.
    """
    if lbe_model.model is None:
        raise RuntimeError("LBE model must be fitted first.")

    probs_test  = lbe_model.predict_proba(X_test)
    probs_train = lbe_model.predict_proba(X_train)

    scores_test_U  = probs_test[:, proba_index][s_test == 0]
    scores_train_N = probs_train[:, proba_index][s_train == 1]

    # orientation check
    if np.mean(scores_test_U) < np.mean(scores_train_N):
        scores_test_U  = 1.0 - scores_test_U
        scores_train_N = 1.0 - scores_train_N

    # robust NU LBE
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

# --------------------------- main experiment ---------------------------

def experiment_residualized_lbe_with_prior(
    name, nsym, lbe_model, unl_mem_ratio=0.5, results_dir="../results", bins=10, device=1
):
    """
    Residualization kept (CONTROL vs CONTROL+MIA), but protocol matches Scripts 1/2:
      - LBEWithPrior scorers
      - prevalence sweep p_ in {0.0, 0.1, ..., 1.0}
      - internal prior used for thresholding (choose_threshold_nu(..., internal_pi))
      - report internal_pi, p_hat_test, p_hat, pi_hat, standard_pi_hat, lowerbound, acc, bacc
      - per-p CSVs (aggregate + full)
    """
    metrics = [
        "acc",
        "bacc",
        "p_hat_test",
        "p_hat",
        "pi_hat",
        "standard_pi_hat",
        "internal_pi",
        "TPR",
        "FPR",
        "J",
        "lowerbound",
    ]

    print("\nMethod: Residualized LBE (Control vs Control+MIA) with INTERNAL PRIOR (Script1/2-style)")

    # Train models once per seed at a reference p (e.g., p=0.5 like Script 1 often does)
    # Here we train per seed with p=0.5; the trained models are reused across p_ sweep.
    models_ctrl = []
    models_comb = []

    for sym in np.arange(0, nsym, 1):
        Xm_tr, Xm_te, y_train, y_test, s_train, Xc_tr, Xc_te = prepare_data(
            name=name, seed=int(sym), p=0.5, unl_mem_ratio=unl_mem_ratio
        )
        np.random.seed(sym)
        lbe_seed(sym)

        # build combined spaces
        Xcm_tr = np.hstack([Xc_tr, Xm_tr])
        Xcm_te = np.hstack([Xc_te, Xm_te])

        # Fit LBEWithPrior on CONTROL-only and COMBINED
        start_time = time.time()
        mdl_ctrl = LBEWithPrior(kind=lbe_model, bins=bins, device=device)
        mdl_ctrl.fit(Xc_tr, s_train)

        mdl_comb = LBEWithPrior(kind=lbe_model, bins=bins, device=device)
        mdl_comb.fit(Xcm_tr, s_train)
        end_time = time.time()
        run_time = end_time - start_time

        mdl_comb._fit_runtime = run_time  # stash for logging
        models_ctrl.append(mdl_ctrl)
        models_comb.append(mdl_comb)

    print("\nTraining finished. Evaluating across prevalence sweep...")

    for p_ in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        records = []

        for sym in np.arange(0, nsym, 1):
            # Rebuild data for this target test prevalence p_
            Xm_tr, Xm_te, y_train, y_test, s_train, Xc_tr, Xc_te = prepare_data(
                name=name, seed=int(sym), p=p_, unl_mem_ratio=unl_mem_ratio
            )
            Xcm_tr = np.hstack([Xc_tr, Xm_tr])
            Xcm_te = np.hstack([Xc_te, Xm_te])

            mdl_ctrl = models_ctrl[sym]
            mdl_comb = models_comb[sym]
            internal_pi = mdl_comb.get_prior()
            run_time = getattr(mdl_comb, "_fit_runtime", np.nan)

            # ---------- residualization (in logit space) ----------
            # Get member-like probs from both scorers
            p_ctrl_tr = mdl_ctrl.predict_proba(Xc_tr)[:, 1]
            p_comb_tr = mdl_comb.predict_proba(Xcm_tr)[:, 1]
            p_ctrl_te = mdl_ctrl.predict_proba(Xc_te)[:, 1]
            p_comb_te = mdl_comb.predict_proba(Xcm_te)[:, 1]

            # logit + OLS: comb ~ ctrl  (remove control-explained component)
            l_ctrl_tr = safe_logit(p_ctrl_tr)
            l_comb_tr = safe_logit(p_comb_tr)
            l_ctrl_te = safe_logit(p_ctrl_te)
            l_comb_te = safe_logit(p_comb_te)

            ols = LinearRegression().fit(l_ctrl_tr.reshape(-1, 1), l_comb_tr)
            l_comb_pred_tr = ols.predict(l_ctrl_tr.reshape(-1, 1))
            l_comb_pred_te = ols.predict(l_ctrl_te.reshape(-1, 1))

            l_resid_tr = l_comb_tr - l_comb_pred_tr
            l_resid_te = l_comb_te - l_comb_pred_te

            prob_train_corrected = np.clip(sigmoid(l_resid_tr), 0.0, 1.0)
            prob_test_corrected  = np.clip(sigmoid(l_resid_te), 0.0, 1.0)

            # Orientation: metrics expect "non-member = 1"
            prob_train_nm = 1.0 - prob_train_corrected
            prob_test_nm  = 1.0 - prob_test_corrected
            y_test_eval = 1 - y_test

            # For internal-prior thresholding, we need neg_scores in the SAME space:
            neg_scores_nm = prob_train_nm[s_train == 1]  # known negatives (non-members)

            # Threshold via internal prior (Script1/2-style)
            best = choose_threshold_nu(prob_test_nm, neg_scores_nm, internal_pi)

            # Debias TEST using TRAIN-derived TPR/FPR at that threshold
            debias_result = debias_target(prob_test_nm, best["thr"], best["TPR"], best["FPR"])

            # Standard PU estimate for reporting (not used for main estimate)
            standard_result = estimate_p_and_debias(prob_test_nm, neg_scores_nm)

            # Compute p_hat_test (Script1/2-style) using raw mdl_comb distributions
            # s_test = all zeros (unlabeled)
            s_test = np.zeros(Xcm_te.shape[0], dtype=int)
            p_hat_test = estimate_p_test_from_trainN(
                lbe_model=mdl_comb,
                X_test=Xcm_te,
                s_test=s_test,
                X_train=Xcm_tr,
                s_train=s_train,
            )

            # Classification metrics using 0.5 over non-member prob
            acc  = accuracy_score(y_test_eval, (prob_test_nm > 0.5).astype(int))
            bacc = balanced_accuracy_score(y_test_eval, (prob_test_nm > 0.5).astype(int))

            results = {
                "method": "resid_LBE_with_prior",
                "run": int(sym) + 1,
                "acc": acc,
                "bacc": bacc,
                "time": run_time,
                "internal_pi": internal_pi,
                "p_hat_test": p_hat_test,                 # LBE NU estimate on test
                "p_hat": debias_result["p_hat"],          # debiased via TPR/FPR and internal prior threshold
                "pi_hat": internal_pi,                    # report internal prior as pi_hat (train)
                "ci_low": debias_result.get("ci_low", np.nan),
                "ci_high": debias_result.get("ci_high", np.nan),
                "threshold": debias_result["threshold"],
                "TPR": debias_result["TPR"],
                "FPR": debias_result["FPR"],
                "J": debias_result["J"],
                "standard_p_hat": standard_result.get("p_hat", np.nan),
                "standard_pi_hat": standard_result.get("pi_hat", np.nan),
                "lowerbound": standard_result.get("lowerbound", np.nan),
            }
            print(results)
            records.append(results)

        df = pd.DataFrame(records)

        # aggregate like Scripts 1/2
        agg = df.groupby("method")[[
            "acc","bacc","p_hat_test","p_hat","pi_hat","standard_pi_hat","internal_pi","TPR","FPR","J","lowerbound"
        ]].agg(["min","mean","std","max"])

        rows = []
        for method in agg.index:
            row = {"method": method}
            for metric in ["acc","bacc","p_hat_test","p_hat","pi_hat","standard_pi_hat","internal_pi","TPR","FPR","J","lowerbound"]:
                row[f"{metric}_mean_std"] = f"{agg[(metric,'mean')][method]:.3f}±{agg[(metric,'std')][method]:.3f}"
            rows.append(row)
        formatted = pd.DataFrame(rows)

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        formatted.to_csv(
            f"{results_dir}/results_lbe_prior_{name}_p={p_}.csv",
            index=False, sep="\t"
        )
        df.to_csv(
            f"{results_dir}/results_lbe_prior_full_{name}_p={p_}.csv",
            index=False, sep="\t"
        )

    # (final printout will reflect the last df in scope)
    print("\nSweep completed.")
    print(f"Mean internal_pi: {df['internal_pi'].mean():.4f} ± {df['internal_pi'].std():.4f}")
    print(f"Mean p_hat:       {df['p_hat'].mean():.4f} ± {df['p_hat'].std():.4f}")
    print(f"Mean p_hat_test:  {df['p_hat_test'].mean():.4f} ± {df['p_hat_test'].std():.4f}")
    print(f"Mean std_pi_hat:  {df['standard_pi_hat'].mean():.4f} ± {df['standard_pi_hat'].std():.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, default="var_24", required=False,
                        help="Model to use. Available options: "
                             "var_16, var_20, var_24 (default), var_30, "
                             "rar_b, rar_l, rar_xl, rar_xxl, "
                             "mar_b, mar_l, mar_h")
    parser.add_argument('-nsym', type=int, required=True, help="Number of iterations/runs")
    parser.add_argument('-lbe_model', type=str, default="LR", required=False,
                        choices=["LR", "MLP"],
                        help="Backbone for LBE: LR (default) or MLP")
    parser.add_argument('-unl_mem_ratio', type=float, default=0.5, required=False,
                        help="Unlabeled members ratio (0..1, default 0.5)")
    parser.add_argument('-results', type=str, default="../results", required=False,
                        help="Directory to save results (default: ../results)")
    parser.add_argument('-bins', type=int, default=10, required=False,
                        help="Number of bins for the LBE model (default: 10)")
    parser.add_argument('-device', type=int, default=1, required=False,
                        help="Device to use for LBE training (default: 1)")
    args = parser.parse_args()

    experiment_residualized_lbe_with_prior(
        name=args.data,
        nsym=args.nsym,
        lbe_model=args.lbe_model,
        unl_mem_ratio=args.unl_mem_ratio,
        results_dir=args.results,
        bins=args.bins,
        device=args.device,
    )

if __name__ == "__main__":
    main()
