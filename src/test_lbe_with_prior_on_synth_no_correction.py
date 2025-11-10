import argparse
import numpy as np
import os
import pandas as pd
import time
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler

from PUBiasCalibration.Models.LBEWithPrior import (
    LBEWithPrior,
    seed,
    _lbe_nu_estimate_p_robust,
)
from PUBiasCalibration.helper_files.pu_metrics import (
    estimate_p_and_debias,
    debias_target,
)


# NOTE: Overall structure, CSV outputs, and the LBEWithPrior workflow are preserved.


def prepare_data(name, seed, p, unl_mem_ratio=0.5, test_size=0.2):
    """
    Prepare data for the experiment (synthetic tp/fp variant).

    Parameters
    ----------
    name : str
        The name of the dataset.
    seed : int
        The random seed.
    p : float
        Target member prevalence in the *test unlabeled* set (0..1).
    unl_mem_ratio : float
        Member ratio in the *training unlabeled* portion (0..1).
    test_size : float
        Test size as a fraction of the constructed training size.

    Returns
    -------
    X_train : (n_train, d) float32
    X_test  : (n_test, d) float32
    y_train : (n_train,) int   # true labels: members=0, nonmembers=1
    y_test  : (n_test,)  int   # true labels: members=0, nonmembers=1
    s_train : (n_train,) int   # observed NU labels: 1 = known negatives (N), 0 = unlabeled (U)
    s_test  : (n_test,)  int   # observed NU labels for test (here: all 0 = unlabeled)
    """
    np.random.seed(seed)

    # choose source file by model prefix
    file_type = "loss" if name.startswith("mar_") else "cfg"

    # --------------------------------------------------------------------------------
    # CHANGED: Load synthetic-style splits using tp/fp + VAL/TRAIN layout (like your
    #          second script). Here:
    #          - VAL_* are used to build labeled non-members (known negatives).
    #          - TRAIN_* are used to build unlabeled and test sets.
    # --------------------------------------------------------------------------------
    val_tp = np.load(
        f"../data/{name}_llm_mia_{file_type}_5k_tp_s0.5_imagenet_val.npz",
        allow_pickle=True,
    )
    X_val_tp = val_tp["data"]

    val_fp = np.load(
        f"../data/{name}_llm_mia_{file_type}_5k_fp_s0.5_imagenet_val.npz",
        allow_pickle=True,
    )
    X_val_fp = val_fp["data"]

    train_tp = np.load(
        f"../data/{name}_llm_mia_{file_type}_5k_tp_s0.5_imagenet_train.npz",
        allow_pickle=True,
    )
    X_train_tp = train_tp["data"]

    train_fp = np.load(
        f"../data/{name}_llm_mia_{file_type}_5k_fp_s0.5_imagenet_train.npz",
        allow_pickle=True,
    )
    X_train_fp = train_fp["data"]

    # shuffle
    X_val_tp = X_val_tp[np.random.permutation(len(X_val_tp))]
    X_val_fp = X_val_fp[np.random.permutation(len(X_val_fp))]
    X_train_tp = X_train_tp[np.random.permutation(len(X_train_tp))]
    X_train_fp = X_train_fp[np.random.permutation(len(X_train_fp))]

    # -----------------------------
    # Train set construction
    # -----------------------------
    # 2000 known negatives (N) + 2000 unlabeled (U with unl_mem_ratio members)

    # --------------------------------------------------------------------------------
    # CHANGED: Build labeled non-members (known negatives s=1) from VAL_tp/VAL_fp mix
    #          so the *observed* labeled negatives can be a mixture controlled by
    #          unl_mem_ratio (mirroring your synthetic setup).
    #          True labels remain: members=0, nonmembers=1
    # --------------------------------------------------------------------------------
    n_lab_from_tp = int(2000 * unl_mem_ratio)
    n_lab_from_fp = 2000 - n_lab_from_tp
    X_train_nonmem = np.concatenate([X_val_tp[:n_lab_from_tp], X_val_fp[:n_lab_from_fp]])

    # true labels: members=0, nonmembers=1
    y_train_nonmem = np.ones(len(X_train_nonmem), dtype=int)  # labeled as non-members
    s_train_nonmem = np.ones(len(X_train_nonmem), dtype=int)  # observed as known negatives (N)

    # --------------------------------------------------------------------------------
    # CHANGED: Build unlabeled set (s=0) from TRAIN_tp (members) and TRAIN_fp (nonmembers)
    # --------------------------------------------------------------------------------
    n_unl_mem = int(2000 * unl_mem_ratio)        # members inside unlabeled
    n_unl_nonmem = 2000 - n_unl_mem              # non-members inside unlabeled

    X_train_unl = np.concatenate([X_train_tp[:n_unl_mem], X_train_fp[:n_unl_nonmem]])

    # true labels: members=0, nonmembers=1
    y_train_unl = np.concatenate(
        [np.zeros(n_unl_mem, dtype=int), np.ones(n_unl_nonmem, dtype=int)]
    )
    s_train_unl = np.zeros(len(X_train_unl), dtype=int)  # unlabeled (U)

    # combine train
    X_train = np.concatenate([X_train_nonmem, X_train_unl])
    y_train = np.concatenate([y_train_nonmem, y_train_unl])
    s_train = np.concatenate([s_train_nonmem, s_train_unl])

    # -----------------------------
    # Test set construction (unlabeled only)
    # -----------------------------
    train_len = len(X_train)
    test_len = int(test_size * train_len)

    # --------------------------------------------------------------------------------
    # CHANGED: Preserve original structure: test set size = test_size * train_len,
    #          but still build test from TRAIN_tp/ TRAIN_fp with target prevalence p.
    # --------------------------------------------------------------------------------
    n_test_unl = test_len
    n_pos_test = int(n_test_unl * p)                 # members inside test unlabeled
    n_unl_test_nonmem = n_test_unl - n_pos_test      # non-members inside test unlabeled

    X_test_unl = np.concatenate(
        [
            X_train_tp[n_unl_mem : n_unl_mem + n_pos_test],
            X_train_fp[n_unl_nonmem : n_unl_nonmem + n_unl_test_nonmem],
        ]
    )

    y_test_unl = np.concatenate(
        [
            np.zeros(n_pos_test, dtype=int),            # members
            np.ones(n_unl_test_nonmem, dtype=int),      # non-members
        ]
    )

    X_test = X_test_unl
    y_test = y_test_unl

    # --------------------------------------------------------------------------------
    # CHANGED: s_test stays all zeros (unlabeled), same as original.
    # --------------------------------------------------------------------------------
    s_test = np.zeros(len(X_test), dtype=int)

    # scale
    X_train = X_train.squeeze(1)
    X_test = X_test.squeeze(1)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, s_train, s_test


def estimate_p_test_from_trainN(
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
      - train known negatives as the N reference (s_train == 1)
    No retraining is done.
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


def experiment_lbe_with_prior(
    name, nsym, lbe_model, unl_mem_ratio=0.5, results_dir="../results", p=0.5, bins=10, device=1
):
    """
    Run the experiment with LBE using internal prior.

    Parameters
    ----------
    name : str
        The name of the dataset.
    nsym : int
        The number of iterations.
    lbe_model : str
        Backbone for LBEWithPrior: 'LR' (Logistic Regression) or 'MLP'.
    p : float
        The probability value.
    unl_mem_ratio : float
        The ratio of members in the unlabeled set.
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
        "p_hat",
        "pi_hat",
        "standard_pi_hat",
        "internal_pi",
        "TPR",
        "FPR",
        "J",
        "lowerbound",
    ]

    print("\n Method: LBE with internal prior")
    models = []
    for sym in np.arange(0, nsym, 1):
        X_train, X_test, y_train, y_test, s_train, s_test = prepare_data(
            name=name, seed=sym, p=p, unl_mem_ratio=unl_mem_ratio
        )
        np.random.seed(sym)
        seed(sym)

        start_time = time.time()
        # Use the LBEWithPrior model (now with selectable backbone)
        model = LBEWithPrior(kind=lbe_model, bins=bins, device=device)
        model.fit(X_train, s_train)
        end_time = time.time()
        run_time = end_time - start_time
        models.append(model)

    print("\n Method: LBE with internal prior - training ended")
    for p_ in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        records = []
        for sym in np.arange(0, nsym, 1):
            X_train, X_test, y_train, y_test, s_train, s_test = prepare_data(
                name=name, seed=sym, p=p_, unl_mem_ratio=unl_mem_ratio
            )
            internal_pi = models[sym].get_prior()

            p_hat_test = estimate_p_test_from_trainN(
                lbe_model=models[sym],
                X_test=X_test,
                s_test=s_test,
                X_train=X_train,
                s_train=s_train,
            )

            prob_y_test = models[sym].predict_proba(X_test)[:, 1]

            # Flip labels
            prob_y_test = 1 - prob_y_test
            y_test = 1 - y_test

            acc = accuracy_score(y_test, np.where(prob_y_test > 0.5, 1, 0))
            bacc = balanced_accuracy_score(y_test, np.where(prob_y_test > 0.5, 1, 0))

            results = {
                "acc": acc,
                "bacc": bacc,
                "time": run_time,
                "internal_pi": internal_pi,
            }

            # Get negative samples (labeled negatives from X_train)
            X_train_neg = X_train[s_train == 1]

            # Get scores for negative samples
            neg_scores = models[sym].predict_proba(X_train_neg)[:, 1]

            # Flip scores to match the flipped labels
            neg_scores = 1 - neg_scores

            # Use the internal prior for debiasing
            # First, find the optimal threshold using the internal prior
            from PUBiasCalibration.helper_files.pu_metrics import choose_threshold_nu

            best = choose_threshold_nu(prob_y_test, neg_scores, internal_pi)

            # Perform debiasing with the internal prior
            debias_result = debias_target(
                prob_y_test, best["thr"], best["TPR"], best["FPR"]
            )

            # Also run the standard estimation for comparison
            standard_result = estimate_p_and_debias(prob_y_test, neg_scores)

            # Add method, run, and results to the results dictionary
            results.update(
                {
                    "method": "lbe_with_internal_prior",
                    "run": sym + 1,
                    "p_hat_test": p_hat_test,  # LBE p - test
                    "p_hat": debias_result["p_hat"],
                    "pi_hat": internal_pi,  # Use internal prior as pi_hat - train
                    "ci_low": debias_result["ci_low"],
                    "ci_high": debias_result["ci_high"],
                    "threshold": debias_result["threshold"],
                    "TPR": debias_result["TPR"],
                    "FPR": debias_result["FPR"],
                    "J": debias_result["J"],
                    "standard_p_hat": standard_result["p_hat"],
                    "standard_pi_hat": standard_result["pi_hat"],
                    "lowerbound": standard_result["lowerbound"],
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

        formatted.to_csv(
            f"{results_dir}/results_lbe_prior_{name}_p={p_}.csv", index=False, sep="\t"
        )
        df.to_csv(
            f"{results_dir}/results_lbe_prior_full_{name}_p={p_}.csv", index=False, sep="\t"
        )

    # Print the true unlabeled members ratio vs. estimated ratio
    print(f"\nTrue unlabeled members ratio: {unl_mem_ratio:.4f}")
    print(f"\nTrue p: {p:.4f}")
    print(
        f"Mean estimated ratio (internal pi): {df['internal_pi'].mean():.4f} ± {df['internal_pi'].std():.4f}"
    )
    print(
        f"Mean estimated ratio (p_hat): {df['p_hat'].mean():.4f} ± {df['p_hat'].std():.4f}"
    )
    print(
        f"Mean estimated ratio (p_hat_test): {df['p_hat_test'].mean():.4f} ± {df['p_hat_test'].std():.4f}"
    )
    print(
        f"Mean estimated ratio (standard pi_hat): {df['standard_pi_hat'].mean():.4f} ± {df['standard_pi_hat'].std():.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-data",
        type=str,
        default="var_24",
        required=False,
        help="Model to use. Available options: "
        "var_16, var_20, var_24 (default), var_30, "
        "rar_b, rar_l, rar_xl, rar_xxl, "
        "mar_b, mar_l, mar_h",
    )
    parser.add_argument(
        "-nsym", type=int, required=True, help="Number of iterations/runs"
    )
    # parser.add_argument(
    #     "-prob", type=float, required=True, help="Probability value (between 0 and 1)"
    # )
    parser.add_argument(
        "-lbe_model",
        type=str,
        default="LR",
        required=False,
        choices=["LR", "MLP"],
        help="Backbone for LBEWithPrior: LR (Logistic Regression, default) or MLP (Multi-layer Perceptron)",
    )
    parser.add_argument(
        "-unl_mem_ratio",
        type=float,
        default=0.5,
        required=False,
        help="Unlabeled members ratio (between 0 and 1, where 1.0 = 2000 members, default: 0.5)",
    )
    parser.add_argument(
        "-results",
        type=str,
        default="../results_synthetic_no_correction",
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
        "-device",
        type=int,
        default=2,
        required=False,
        help="Device to use for LBE training (default: 1)",
    )
    args = parser.parse_args()

    experiment_lbe_with_prior(
        args.data,
        args.nsym,
        args.lbe_model,
        args.unl_mem_ratio,
        args.results,
        bins=args.bins,
        device=args.device,
    )


if __name__ == "__main__":
    main()
