import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_score, recall_score, roc_curve, \
    auc, precision_recall_curve
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler

import PUBiasCalibration.Models.LBE as lbe
import PUBiasCalibration.Models.PGlin as pgl
import PUBiasCalibration.Models.PUSB as pusb
import PUBiasCalibration.Models.SAREM as sarem
import PUBiasCalibration.Models.basic as basic
import PUBiasCalibration.Models.threshold as threshold
import PUBiasCalibration.helper_files.km as km
from PUBiasCalibration.Models.LBE import LBE
from PUBiasCalibration.Models.PGlin import PUGerych
from PUBiasCalibration.Models.SAREM import SAREM
from PUBiasCalibration.Models.basic import PUbasic
from PUBiasCalibration.Models.threshold import PUthreshold
from PUBiasCalibration.helper_files.pu_metrics import estimate_p_and_debias


def prepare_data(name, seed, p, unl_mem_ratio=0.5, test_size=0.2):
    np.random.seed(seed)

    # Determine if we should use "cfg" or "loss" based on the model prefix
    file_type = "loss" if name.startswith("mar_") else "cfg"

    # -----------------------------
    # Load data
    # -----------------------------
    # Training labeled non-members come from VAL files (mixture)
    val_tp = np.load(f"../data/{name}_llm_mia_{file_type}_5k_tp_s0.5_imagenet_val.npz", allow_pickle=True)
    X_val_tp = val_tp["data"]
    val_fp = np.load(f"../data/{name}_llm_mia_{file_type}_5k_fp_s0.5_imagenet_val.npz", allow_pickle=True)
    X_val_fp = val_fp["data"]

    # Unlabeled and test samples come from TRAIN files
    train_tp = np.load(f"../data/{name}_llm_mia_{file_type}_5k_tp_s0.5_imagenet_train.npz", allow_pickle=True)
    X_train_tp = train_tp["data"]
    train_fp = np.load(f"../data/{name}_llm_mia_{file_type}_5k_fp_s0.5_imagenet_train.npz", allow_pickle=True)
    X_train_fp = train_fp["data"]

    X_val_tp = X_val_tp[np.random.permutation(len(X_val_tp))]
    X_val_fp = X_val_fp[np.random.permutation(len(X_val_fp))]
    X_train_tp = X_train_tp[np.random.permutation(len(X_train_tp))]
    X_train_fp = X_train_fp[np.random.permutation(len(X_train_fp))]

    # -----------------------------
    # Train set construction
    # -----------------------------
    # 2000 labeled non-members + 2000 unlabeled (with varying member ratio)

    # ----- Labeled non-members (y=1, s=1) -----
    n_lab_from_tp = int(2000 * unl_mem_ratio)
    n_lab_from_fp = 2000 - n_lab_from_tp
    X_train_nonmem = np.concatenate([
        X_val_tp[:n_lab_from_tp],
        X_val_fp[:n_lab_from_fp]
    ])
    y_train_nonmem = np.ones(len(X_train_nonmem), dtype=int)  # labeled as non-members
    s_train_nonmem = np.ones(len(X_train_nonmem), dtype=int)  # labeled samples

    # ----- Unlabeled (s=0) -----
    # Calculate number of unlabeled members based on ratio (2000 = 100%)
    n_unl_mem = int(2000 * unl_mem_ratio)
    n_unl_nonmem = 2000 - n_unl_mem

    X_train_unl = np.concatenate([
        X_train_tp[:n_unl_mem],
        X_train_fp[:n_unl_nonmem]
    ])

    # True labels: nonmembers=1, members=0
    y_train_unl = np.concatenate([
        np.zeros(n_unl_mem, dtype=int),       # true members
        np.ones(n_unl_nonmem, dtype=int)      # true nonmembers
    ])
    s_train_unl = np.zeros(len(X_train_unl), dtype=int)        # unlabeled samples

    X_train = np.concatenate([X_train_nonmem, X_train_unl])
    y_train = np.concatenate([y_train_nonmem, y_train_unl])
    s_train = np.concatenate([s_train_nonmem, s_train_unl])

    train_len = len(X_train)
    test_len = int(test_size * train_len)

    n_test_unl = 2000
    n_pos_test = int(n_test_unl * p)
    n_unl_test_nonmem = n_test_unl - n_pos_test

    # -----------------------------
    # Test set construction
    # -----------------------------
    # X_test_nonmem = X_nonmem[2000 + n_unl_nonmem : 2000 + n_unl_nonmem + n_test_nonmem]

    X_test_unl = np.concatenate([
        X_train_tp[n_unl_mem : n_unl_mem + n_pos_test],
        X_train_fp[n_unl_nonmem :
                   n_unl_nonmem + n_unl_test_nonmem]
    ])

    # y_test_nonmem = np.ones(len(X_test_nonmem), dtype=int)
    y_test_unl = np.concatenate([
        np.zeros(n_pos_test, dtype=int),
        np.ones(n_unl_test_nonmem, dtype=int)
    ])

    X_test = X_test_unl
    y_test = y_test_unl

    X_train = X_train.squeeze(1)
    X_test = X_test.squeeze(1)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, s_train


def experiment_lr(name, nsym, p, unl_mem_ratio=0.5, results_dir="../results"):
    records = []
    methods = ['threshold', 'sar-em', 'pusb', 'pglin', 'lbe', 'oracle', 'dummy', 'threshold_balanced']
    metrics = ["acc", "p_hat", "pi_hat", "TPR", "FPR", "J", "lowerbound"]#["acc", "bacc", "rec", "prec", "f1", "tpr", "tnr", "roc_auc", "pr_auc" , "time"]
    for method in methods:
        print('\n Method:', method)
        for sym in np.arange(0, nsym, 1):

            X_train, X_test, y_train, y_test, s_train = prepare_data(name=name, seed=sym, p=p, unl_mem_ratio=unl_mem_ratio)
            np.random.seed(sym)
            km.seed(sym)
            pusb.seed(sym)
            lbe.seed(sym)
            pgl.seed(sym)
            sarem.seed(sym)
            threshold.seed(sym)
            basic.seed(sym)

            start_time = time.time()
            if method == 'oracle':
                model = PUbasic()
                model.fit(X_train, y_train)
            elif method == 'dummy':
                model = PUbasic()
                model.fit(X_train, s_train)
            else:
                if method == 'threshold':
                    model = PUthreshold()
                elif method == 'sar-em':
                    model = SAREM()
                elif method == 'lbe':
                    model = LBE()
                elif method == 'pglin':
                    model = PUGerych()
                model.fit(X_train, s_train)
            end_time = time.time()
            run_time = end_time - start_time

            prob_y_test = model.predict_proba(X_test)[:, 1]

            #flip_labels
            prob_y_test = 1 - prob_y_test
            y_test = 1 - y_test

            acc = accuracy_score(y_test, np.where(prob_y_test > 0.5, 1, 0))
            # rec = recall_score(y_test, np.where(prob_y_test > 0.5, 1, 0))
            # prec = precision_score(y_test, np.where(prob_y_test > 0.5, 1, 0))
            # f1 = f1_score(y_test, np.where(prob_y_test > 0.5, 1, 0))
            # tpr = np.count_nonzero(np.where(prob_y_test > 0.5, 1, 0)[y_test == 1] == 1) / np.count_nonzero(y_test == 1)
            # tnr = np.count_nonzero(np.where(prob_y_test > 0.5, 1, 0)[y_test == 0] == 0) / np.count_nonzero(y_test == 0)
            # fpr_thr, tpr_thr, thr = roc_curve(y_test, prob_y_test, pos_label=1)
            # roc_auc = auc(fpr_thr, tpr_thr)
            # prec_thr, recall_thr, thr = precision_recall_curve(y_test, prob_y_test)
            # pr_auc = auc(recall_thr, prec_thr)
            bacc = balanced_accuracy_score(y_test, np.where(prob_y_test > 0.5, 1, 0))

            results = {
                "acc": acc,
                "bacc": bacc,
            #     "rec": rec,
            #     "prec": prec,
            #     "f1": f1,
            #     "tpr": tpr,
            #     "tnr": tnr,
            #     "roc_auc": roc_auc,
            #     "pr_auc": pr_auc,
                "time": run_time,
            }
            #estimate p_hat here
            # Get negative samples (labeled negatives from X_train)
            X_train_neg = X_train[s_train == 1]

            # Get scores for negative samples
            neg_scores = model.predict_proba(X_train_neg)[:, 1]

            # Flip scores to match the flipped labels
            neg_scores = 1 - neg_scores

            # Estimate p_hat and perform debiasing (includes lowerbound calculation)
            p_hat_results = estimate_p_and_debias(prob_y_test, neg_scores)

            # Add method, run, and p_hat results to the results dictionary
            results.update({
                "method": method, 
                "run": sym+1,
                "p_hat": p_hat_results["p_hat"],
                "pi_hat": p_hat_results["pi_hat"],
                "ci_low": p_hat_results["ci_low"],
                "ci_high": p_hat_results["ci_high"],
                "threshold": p_hat_results["threshold"],
                "TPR": p_hat_results["TPR"],
                "FPR": p_hat_results["FPR"],
                "J": p_hat_results["J"],
                "lowerbound": p_hat_results["lowerbound"]
            })
            print(results)
            records.append(results)

            #
    df = pd.DataFrame(records)

    agg = df.groupby("method")[metrics].agg(["min", "mean", "std", "max"])

    # Step 3: Format mean ± std
    rows = []
    for method in agg.index:
        row = {"method": method}
        for metric in metrics:
            # row[f"{metric}_min"] = agg[(metric, "min")][method]
            row[f"{metric}_mean_std"] = f"{agg[(metric, 'mean')][method]:.3f}±{agg[(metric, 'std')][method]:.3f}"
            # row[f"{metric}_max"] = agg[(metric, "max")][method]
        rows.append(row)

    formatted = pd.DataFrame(rows)

    # Create results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    formatted.to_csv(f"{results_dir}/results_agg_{name}_p={p}.csv", index=False, sep="\t")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, default="var_24", required=False,
                        help="Model to use. Available options: "
                             "var_16, var_20, var_24 (default), var_30, "
                             "rar_b, rar_l, rar_xl, rar_xxl, "
                             "mar_b, mar_l, mar_h")
    parser.add_argument('-nsym', type=int, required=True,
                        help="Number of iterations/runs")
    parser.add_argument('-prob', type=float, required=True,
                        help="Probability value (between 0 and 1)")
    parser.add_argument('-unl_mem_ratio', type=float, default=0.5, required=False,
                        help="Unlabeled members ratio (between 0 and 1, where 1.0 = 2000 members, default: 0.5)")
    parser.add_argument('-results', type=str, default="../results", required=False,
                        help="Directory to save results (default: ../results)")
    args = parser.parse_args()

    experiment_lr(args.data, args.nsym, args.prob, args.unl_mem_ratio, args.results)


if __name__ == "__main__":
    main()
