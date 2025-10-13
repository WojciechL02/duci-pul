import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import time
import PUBiasCalibration.helper_files.km as km
import pandas as pd
from PUBiasCalibration.helper_files.pu_metrics import estimate_p_and_debias

import PUBiasCalibration.Models.PUSB as pusb
from PUBiasCalibration.Models.PUSB import PUSB
import PUBiasCalibration.Models.LBE as lbe
from PUBiasCalibration.Models.LBE import LBE
import PUBiasCalibration.Models.PGlin as pgl
from PUBiasCalibration.Models.PGlin import PUGerych
from PUBiasCalibration.helper_files.utils import make_binary_class, sigmoid
import PUBiasCalibration.Models.basic as basic
from PUBiasCalibration.Models.basic import PUbasic
import PUBiasCalibration.Models.SAREM as sarem
from PUBiasCalibration.Models.SAREM import SAREM
import PUBiasCalibration.Models.threshold as threshold
from PUBiasCalibration.Models.threshold import PUthreshold
from PUBiasCalibration.Models.PUe import PUe
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_score, recall_score, roc_curve, auc, precision_recall_curve


def prepare_data(name, seed, p, test_size=0.2):
    np.random.seed(seed)

    members = np.load(f"../data/{name}_llm_mia_cfg_5k_real_imagenet_train.npz", allow_pickle=True)
    X_mem = members["data"]
    nonmembers = np.load(f"../data/{name}_llm_mia_cfg_5k_real_imagenet_val.npz", allow_pickle=True)
    X_nonmem = nonmembers["data"]

    X_mem = X_mem[np.random.permutation(len(X_mem))]
    X_nonmem = X_nonmem[np.random.permutation(len(X_nonmem))]

    # -----------------------------
    # Train set construction
    # -----------------------------
    # 2000 nonmembers + 1000 unlabeled (p% members)
    X_train_nonmem = X_nonmem[:2000]

    n_unl_mem = int(1000 * p)  # members in unlabeled
    n_unl_nonmem = 1000 - n_unl_mem

    X_train_unl = np.concatenate([
        X_mem[:n_unl_mem],
        X_nonmem[2000:2000 + n_unl_nonmem]
    ])

    # True labels: nonmembers=1, members=0
    y_train_nonmem = np.ones(len(X_train_nonmem), dtype=int)
    y_train_unl = np.concatenate([
        np.zeros(n_unl_mem, dtype=int),       # true members
        np.ones(n_unl_nonmem, dtype=int)      # true nonmembers
    ])

    X_train = np.concatenate([X_train_nonmem, X_train_unl])
    y_train = np.concatenate([y_train_nonmem, y_train_unl])

    # PU labels: only unlabeled members are marked as 1 (positives)
    s_train = np.concatenate([
        np.ones(len(X_train_nonmem), dtype=int),      # labeled negatives
        np.zeros(len(X_train_unl), dtype=int),        # all unlabeled
    ])

    train_len = len(X_train)
    test_len = int(test_size * train_len)

    # Keep same ratio: 2:1 (nonmembers : unlabeled)
    # n_test_nonmem = int(test_len * (2 / 3))
    n_test_unl = test_len  # - n_test_nonmem
    n_pos_test = int(n_test_unl * p)
    n_unl_test_nonmem = n_test_unl - n_pos_test

    # -----------------------------
    # Test set construction
    # -----------------------------
    # X_test_nonmem = X_nonmem[2000 + n_unl_nonmem : 2000 + n_unl_nonmem + n_test_nonmem]

    X_test_unl = np.concatenate([
        X_mem[n_unl_mem : n_unl_mem + n_pos_test],
        X_nonmem[2000 + n_unl_nonmem :
                 2000 + n_unl_nonmem + n_unl_test_nonmem]
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


def experiment_lr(name, nsym, p):
    records = []
    methods = ['threshold', 'sar-em', 'pusb', 'pglin', 'lbe', 'oracle', 'dummy', 'threshold_balanced']
    metrics = ["acc", "p_hat", "pi_hat", "TPR", "FPR", "J"]#["acc", "bacc", "rec", "prec", "f1", "tpr", "tnr", "roc_auc", "pr_auc" , "time"]
    for method in methods:
        print('\n Method:', method)
        for sym in np.arange(0, nsym, 1):

            X_train, X_test, y_train, y_test, s_train = prepare_data(name=name, seed=sym, p=p)
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

            # Estimate p_hat and perform debiasing
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
                "J": p_hat_results["J"]
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

    formatted.to_csv(f"../results/results_agg_{name}_p={p}.csv", index=False, sep="\t")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, default="var_24", required=False)
    parser.add_argument('-nsym', type=int, required=True)
    parser.add_argument('-prob', type=float, required=True)
    args = parser.parse_args()

    experiment_lr(args.data, args.nsym, args.prob)


if __name__ == "__main__":
    main()
