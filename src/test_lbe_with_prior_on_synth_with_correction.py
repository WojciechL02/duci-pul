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

import PUBiasCalibration.Models.LBE as lbe  # seed helper
from PUBiasCalibration.Models.LBE import LBE  # ### CHANGED: use LBE as the scorer
from PUBiasCalibration.helper_files.pu_metrics import estimate_p_and_debias

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# --- helpers --------------------------------------------------------------------

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


# --- data prep (synthetic tp/fp + control), faithful to your script2 --------------

def prepare_data(name, seed, p, unl_mem_ratio=0.5, test_size=0.2):
    np.random.seed(seed)

    file_type = "loss" if name.startswith("mar_") else "cfg"

    # MIA features
    val_tp = np.load(f"../data/{name}_llm_mia_{file_type}_5k_tp_s0.5_imagenet_val.npz", allow_pickle=True)
    X_val_tp = val_tp["data"]
    val_fp = np.load(f"../data/{name}_llm_mia_{file_type}_5k_fp_s0.5_imagenet_val.npz", allow_pickle=True)
    X_val_fp = val_fp["data"]
    train_tp = np.load(f"../data/{name}_llm_mia_{file_type}_5k_tp_s0.5_imagenet_train.npz", allow_pickle=True)
    X_train_tp = train_tp["data"]
    train_fp = np.load(f"../data/{name}_llm_mia_{file_type}_5k_fp_s0.5_imagenet_train.npz", allow_pickle=True)
    X_train_fp = train_fp["data"]

    # shuffle
    X_val_tp = X_val_tp[np.random.permutation(len(X_val_tp))]
    X_val_fp = X_val_fp[np.random.permutation(len(X_val_fp))]
    X_train_tp = X_train_tp[np.random.permutation(len(X_train_tp))]
    X_train_fp = X_train_fp[np.random.permutation(len(X_train_fp))]

    # CONTROL features
    control_prefix = "ControlSEnsemble"
    X_ctrl_train_fp = load_features(f"../data/{control_prefix}_5k_fp_s0.5_imagenet_train.npz")
    X_ctrl_train_tp = load_features(f"../data/{control_prefix}_5k_tp_s0.5_imagenet_train.npz")
    X_ctrl_val_fp = load_features(f"../data/{control_prefix}_5k_fp_s0.5_imagenet_val.npz")
    X_ctrl_val_tp = load_features(f"../data/{control_prefix}_5k_tp_s0.5_imagenet_val.npz")

    X_ctrl_all = np.concatenate([X_ctrl_train_tp, X_ctrl_train_fp, X_ctrl_val_tp, X_ctrl_val_fp], axis=0)
    np.random.shuffle(X_ctrl_all)

    # -----------------------------
    # Train set construction
    # -----------------------------
    n_lab_from_tp = int(2000 * unl_mem_ratio)
    n_lab_from_fp = 2000 - n_lab_from_tp
    X_train_nonmem = np.concatenate([X_val_tp[:n_lab_from_tp], X_val_fp[:n_lab_from_fp]])
    y_train_nonmem = np.ones(len(X_train_nonmem), dtype=int)     # nonmembers=1
    s_train_nonmem = np.ones(len(X_train_nonmem), dtype=int)     # known negatives

    n_unl_mem = int(2000 * unl_mem_ratio)
    n_unl_nonmem = 2000 - n_unl_mem
    X_train_unl = np.concatenate([X_train_tp[:n_unl_mem], X_train_fp[:n_unl_nonmem]])
    y_train_unl = np.concatenate([np.zeros(n_unl_mem, dtype=int), np.ones(n_unl_nonmem, dtype=int)])
    s_train_unl = np.zeros(len(X_train_unl), dtype=int)          # unlabeled

    X_train = np.concatenate([X_train_nonmem, X_train_unl])
    y_train = np.concatenate([y_train_nonmem, y_train_unl])
    s_train = np.concatenate([s_train_nonmem, s_train_unl])

    # -----------------------------
    # Test set (unlabeled only), size fixed like script2 (2000), or keep original rule?
    # The example you pasted uses fixed 2000. We'll keep that for fidelity. ### CHANGED
    # -----------------------------
    n_test_unl = 2000
    n_pos_test = int(n_test_unl * p)
    n_unl_test_nonmem = n_test_unl - n_pos_test

    X_test_unl = np.concatenate([
        X_train_tp[n_unl_mem: n_unl_mem + n_pos_test],
        X_train_fp[n_unl_nonmem: n_unl_nonmem + n_unl_test_nonmem]
    ])
    y_test_unl = np.concatenate([
        np.zeros(n_pos_test, dtype=int),
        np.ones(n_unl_test_nonmem, dtype=int)
    ])
    X_test = X_test_unl
    y_test = y_test_unl

    # --- scale MIA features ---
    X_train = X_train.squeeze(1)
    X_test = X_test.squeeze(1)
    scaler_mia = MinMaxScaler()
    X_train = scaler_mia.fit_transform(X_train)
    X_test = scaler_mia.transform(X_test)

    # --- scale & align CONTROL features ---
    X_ctrl_all = X_ctrl_all.squeeze(1)
    scaler_ctrl = MinMaxScaler()
    X_ctrl_all = scaler_ctrl.fit_transform(X_ctrl_all)
    X_ctrl_train = X_ctrl_all[:len(X_train)]
    X_ctrl_test = X_ctrl_all[len(X_train):len(X_train) + len(X_test)]

    return X_train, X_test, y_train, y_test, s_train, X_ctrl_train, X_ctrl_test


# --- main experiment (two LBE scorers + residualization; debias on corrected probs) ---

def experiment_lbe_two_scorers(name, nsym, p, unl_mem_ratio=0.5, results_dir="../results"):
    records = []
    metrics = ["acc", "p_hat", "pi_hat", "TPR", "FPR", "J", "lowerbound", "bacc"]

    print("\nMethod: LBE(control-only) vs LBE(control+MIA) → residualized; debias on corrected probs")
    for sym in np.arange(0, nsym, 1):
        # load data
        X_mia_train, X_mia_test, y_train, y_test, s_train, X_ctrl_train, X_ctrl_test = prepare_data(
            name=name, seed=sym, p=p, unl_mem_ratio=unl_mem_ratio
        )

        # seeds
        np.random.seed(sym)
        lbe.seed(sym)

        start_time = time.time()

        # --- standardize feature spaces used for LBE scorers (match distributions) --- ### CHANGED
        sc_ctrl = StandardScaler().fit(X_ctrl_train)
        sc_mia = StandardScaler().fit(X_mia_train)

        Xc_tr = sc_ctrl.transform(X_ctrl_train)
        Xc_te = sc_ctrl.transform(X_ctrl_test)
        Xm_tr = sc_mia.transform(X_mia_train)
        Xm_te = sc_mia.transform(X_mia_test)

        Xcm_tr = np.hstack([Xc_tr, Xm_tr])  # control + MIA (faithful to Option A)
        Xcm_te = np.hstack([Xc_te, Xm_te])

        # --- LBE scorer #1: CONTROL-only --- ### CHANGED
        lbe_ctrl = LBE()                 # ensure you use same hyperparams for both if configurable
        lbe_ctrl.fit(Xc_tr, s_train)     # fit with NU labels (known negatives vs unlabeled)
        p_ctrl_train = lbe_ctrl.predict_proba(Xc_tr)[:, 1]  # member-like prob
        p_ctrl_test  = lbe_ctrl.predict_proba(Xc_te)[:, 1]

        # --- LBE scorer #2: CONTROL + MIA --- ### CHANGED
        lbe_comb = LBE()
        lbe_comb.fit(Xcm_tr, s_train)
        p_comb_train = lbe_comb.predict_proba(Xcm_tr)[:, 1]
        p_comb_test  = lbe_comb.predict_proba(Xcm_te)[:, 1]

        # --- Residualization in logit space (comb ~ ctrl) --- ### CHANGED
        l_ctrl_tr = safe_logit(p_ctrl_train)
        l_comb_tr = safe_logit(p_comb_train)
        l_ctrl_te = safe_logit(p_ctrl_test)
        l_comb_te = safe_logit(p_comb_test)

        ols = LinearRegression().fit(l_ctrl_tr.reshape(-1, 1), l_comb_tr)
        l_comb_pred_te  = ols.predict(l_ctrl_te.reshape(-1, 1))
        l_comb_pred_tr  = ols.predict(l_ctrl_tr.reshape(-1, 1))

        l_resid_te = l_comb_te - l_comb_pred_te
        l_resid_tr = l_comb_tr - l_comb_pred_tr

        prob_y_test_corrected  = np.clip(sigmoid(l_resid_te), 0.0, 1.0)    # member-like (corrected)
        prob_y_train_corrected = np.clip(sigmoid(l_resid_tr), 0.0, 1.0)

        # --- Evaluation + debiasing strictly on corrected probs (no mixing) --- ### CHANGED
        # flip orientation to nonmember=1 for metrics & debias (as in your scripts)
        prob_y_test = 1 - prob_y_test_corrected
        y_test_eval = 1 - y_test

        acc  = accuracy_score(y_test_eval, np.where(prob_y_test > 0.5, 1, 0))
        bacc = balanced_accuracy_score(y_test_eval, np.where(prob_y_test > 0.5, 1, 0))

        # negative reference scores (known negatives in train) in the SAME corrected space
        neg_scores = 1 - prob_y_train_corrected[s_train == 1]  # flip to nonmember=1

        # prevalence & debiasing on corrected scores
        p_hat_results = estimate_p_and_debias(prob_y_test, neg_scores)

        end_time = time.time()
        run_time = end_time - start_time

        results = {
            "method": "lbe_ctrl_vs_comb_residual",
            "run": int(sym) + 1,
            "acc": acc,
            "bacc": bacc,
            "p_hat": p_hat_results["p_hat"],
            "pi_hat": p_hat_results["pi_hat"],
            "ci_low": p_hat_results["ci_low"],
            "ci_high": p_hat_results["ci_high"],
            "threshold": p_hat_results["threshold"],
            "TPR": p_hat_results["TPR"],
            "FPR": p_hat_results["FPR"],
            "J": p_hat_results["J"],
            "lowerbound": p_hat_results["lowerbound"],
            "time": run_time,
        }
        print(results)
        records.append(results)

    # aggregate & save
    df = pd.DataFrame(records)
    agg = df.groupby("method")[metrics].agg(["min", "mean", "std", "max"])

    rows = []
    for method in agg.index:
        row = {"method": method}
        for metric in metrics:
            row[f"{metric}_mean_std"] = f"{agg[(metric, 'mean')][method]:.3f}±{agg[(metric, 'std')][method]:.3f}"
        rows.append(row)
    formatted = pd.DataFrame(rows)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    formatted.to_csv(f"{results_dir}/results_lbe2_residual_agg_{name}_p={p}.csv", index=False, sep="\t")
    df.to_csv(f"{results_dir}/results_lbe2_residual_full_{name}_p={p}.csv", index=False, sep="\t")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, default="var_24", required=False,
                        help="Model to use. Available options: "
                             "var_16, var_20, var_24 (default), var_30, "
                             "rar_b, rar_l, rar_xl, rar_xxl, "
                             "mar_b, mar_l, mar_h")
    parser.add_argument('-nsym', type=int, required=True, help="Number of iterations/runs")
    parser.add_argument('-prob', type=float, required=True, help="Probability value (between 0 and 1)")
    parser.add_argument('-unl_mem_ratio', type=float, default=0.5, required=False,
                        help="Unlabeled members ratio (between 0 and 1, where 1.0 = 2000 members, default: 0.5)")
    parser.add_argument('-results', type=str, default="../results", required=False,
                        help="Directory to save results (default: ../results)")
    args = parser.parse_args()

    experiment_lbe_two_scorers(args.data, args.nsym, args.prob, args.unl_mem_ratio, args.results)


if __name__ == "__main__":
    main()
