import argparse
import numpy as np
import numpy as np
import os
import pandas as pd
import time
from pathlib import Path
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

from PUBiasCalibration.Models.LBEWithPrior import (
    LBEWithPrior,
    seed,
    _lbe_nu_estimate_p_robust,
)


def prepare_data(name, seed, p, test_len=4000):
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
    X_train : (n_train, d) float32
    X_test  : (n_test, d) float32
    y_train : (n_train,) int   # true labels: members=0, nonmembers=1
    y_test  : (n_test,)  int   # true labels: members=0, nonmembers=1
    s_train : (n_train,) int   # observed NU labels: 1 = known negatives (N), 0 = unlabeled (U)
    s_test  : (n_test,)  int   # observed NU labels for test (here: all 0 = unlabeled)
    """
    np.random.seed(seed)

    folder = Path("../data")
    pattern1 = f"{name}_*real_*train.npz"
    pattern2 = f"{name}_*real_*val.npz"

    matches1 = list(folder.glob(pattern1))
    matches2 = list(folder.glob(pattern2))
    if not matches1 or not matches2:
        raise FileNotFoundError(
            f"No files found for patterns: {pattern1} or {pattern2} in {folder}"
        )

    members = np.load(
        matches1[0],
        allow_pickle=True,
    )
    X_mem = members["data"]
    nonmembers = np.load(
        matches2[0],
        allow_pickle=True,
    )
    X_nonmem = nonmembers["data"]

    # shuffle
    X_mem = X_mem[np.random.permutation(len(X_mem))]
    X_nonmem = X_nonmem[np.random.permutation(len(X_nonmem))]

    # # -----------------------------
    # # Train set construction
    # # -----------------------------
    # # 2000 known negatives (N) + 2000 unlabeled (U with unl_mem_ratio members)
    # X_train_nonmem = X_nonmem[:2000]
    #
    # n_unl_mem = int(2000 * unl_mem_ratio)  # members inside unlabeled
    # n_unl_nonmem = 2000 - n_unl_mem  # non-members inside unlabeled
    #
    # X_train_unl = np.concatenate(
    #     [X_mem[:n_unl_mem], X_nonmem[2000 : 2000 + n_unl_nonmem]]
    # )
    #
    # # true labels: members=0, nonmembers=1
    # y_train_nonmem = np.ones(len(X_train_nonmem), dtype=int)
    # y_train_unl = np.concatenate(
    #     [np.zeros(n_unl_mem, dtype=int), np.ones(n_unl_nonmem, dtype=int)]
    # )
    #
    # X_train = np.concatenate([X_train_nonmem, X_train_unl])
    # y_train = np.concatenate([y_train_nonmem, y_train_unl])
    #
    # # NU observed labels: 1 = known negatives (N), 0 = unlabeled (U)
    # s_train = np.concatenate(
    #     [
    #         np.ones(len(X_train_nonmem), dtype=int),  # N
    #         np.zeros(len(X_train_unl), dtype=int),  # U
    #     ]
    # )

    # -----------------------------
    # Test set construction (unlabeled only)
    # -----------------------------
    # train_len = len(X_train)
    X_test_nonmem = X_nonmem[:2000]
    n_test_unl = test_len - 2000
    n_pos_test = int(n_test_unl * p)  # members inside test unlabeled
    n_unl_test_nonmem = n_test_unl - n_pos_test  # non-members inside test unlabeled

    X_test_unl = np.concatenate(
        [
            X_test_nonmem,
            X_mem[:n_pos_test],
            X_nonmem[2000:2000+n_unl_test_nonmem],
        ]
    )

    y_test_unl = np.concatenate(
        [
            np.ones(2000, dtype=int), #NM_labeled
            np.zeros(n_pos_test, dtype=int),  # Umembers
            np.ones(n_unl_test_nonmem, dtype=int),  # Unon-members
        ]
    )

    X_test = X_test_unl
    y_test = y_test_unl

    s_test = np.concatenate(
        [
            np.ones(2000, dtype=int), #NM_labeled
            np.zeros(n_pos_test, dtype=int),  # Umembers
            np.zeros(n_unl_test_nonmem, dtype=int),  # Unon-members
        ]
    )


    # scale
    # X_train = X_train.squeeze(1)
    X_test = X_test.squeeze(1)
    scaler = MinMaxScaler()
    # X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    return X_test, y_test, s_test


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


def experiment_lbe_with_prior(
    name,
    nsym,
    lbe_model,
    results_dir="../results",
    p=0.5,
    bins=10,
    device=1,
):
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
        "p_hat_through_scores_and_debiased"
    ]

    print("\n Method: LBE with internal prior")
    records = []
    for sym in np.arange(0, nsym, 1):
        X_test, y_test, s_test = prepare_data(
            name=name, seed=sym, p=p
        )
        np.random.seed(sym)
        seed(sym)

        start_time = time.time()
        # Use the LBEWithPrior model
        model = LBEWithPrior(kind=lbe_model, bins=bins, device=device)
        model.fit(X_test, s_test)
        end_time = time.time()
        run_time = end_time - start_time

        internal_pi = model.get_prior()

        # Use for correction only
        p_hat_test = estimate_p_test(
            lbe_model=model,
            X_test=X_test[2000:],
            s_test=s_test[2000:],
            X_train=X_test[:2000], # for known NM or generated
            s_train=s_test[:2000], # for known NM or generated
        )

        prob_y_test = model.predict_proba(X_test)[:, 1]

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
            "p_hat_through_scores_and_debiased": p_hat_test,
        }

        # # Get negative samples (labeled negatives from X_train)
        # X_train_neg = X_train[s_train == 1]
        #
        # # Get scores for negative samples
        # neg_scores = model.predict_proba(X_train_neg)[:, 1]
        #
        # # Flip scores to match the flipped labels
        # neg_scores = 1 - neg_scores
        #
        # # First, find the optimal threshold using the internal prior
        # from PUBiasCalibration.helper_files.pu_metrics import choose_threshold_nu
        #
        # best = choose_threshold_nu(prob_y_test, neg_scores, internal_pi)
        #
        # # Perform debiasing with the internal prior
        # debias_result = debias_target(
        #     prob_y_test, best["thr"], best["TPR"], best["FPR"]
        # )

        # Also run the standard estimation for comparison
        # standard_result = estimate_p_and_debias(prob_y_test, neg_scores)

        # Add method, run, and results to the results dictionary
        # results.update(
        #     {
        #         "method": "lbe_with_internal_prior",
        #         "run": sym + 1,
        #         "p_hat_test": p_hat_test,  # LBE p - test
        #         "p_hat": debias_result["p_hat"],
        #         "pi_hat": internal_pi,  # Use internal prior as pi_hat - train
        #         "ci_low": debias_result["ci_low"],
        #         "ci_high": debias_result["ci_high"],
        #         "threshold": debias_result["threshold"],
        #         "TPR": debias_result["TPR"],
        #         "FPR": debias_result["FPR"],
        #         "J": debias_result["J"],
        #         "standard_p_hat": standard_result["p_hat"],
        #         "standard_pi_hat": standard_result["pi_hat"],
        #         "lowerbound": standard_result["lowerbound"],
        #     }
        # )

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
        f"{results_dir}/results_lbe_prior_{model_name}_p={p}.csv",
        index=False,
        sep="\t",
    )
    df.to_csv(
        f"{results_dir}/results_lbe_prior_full_{model_name}_p={p}.csv",
        index=False,
        sep="\t",
    )

    # Print the true unlabeled members ratio vs. estimated ratio
    print(f"\nTrue p: {p:.4f}")
    # print(
    #     f"Mean estimated ratio (internal pi): {df['internal_pi'].mean():.4f} ± {df['internal_pi'].std():.4f}"
    # )
    # print(
    #     f"Mean estimated ratio (p_hat): {df['p_hat'].mean():.4f} ± {df['p_hat'].std():.4f}"
    # )
    print(
        f"Mean estimated ratio (p_hat_test): {df['p_hat_test'].mean():.4f} ± {df['p_hat_test'].std():.4f}"
    )
    print(
        f"Mean estimated ratio (p_hat_through_scores_and_debiased): {df['p_hat_through_scores_and_debiased'].mean():.4f} ± {df['p_hat_through_scores_and_debiased'].std():.4f}"
    )
    # print(
    #     f"Mean estimated ratio (standard pi_hat): {df['standard_pi_hat'].mean():.4f} ± {df['standard_pi_hat'].std():.4f}"
    # )


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
        "-unl_mem_ratio",
        type=float,
        default=0.5,
        required=False,
        help="Unlabeled members ratio (between 0 and 1, where 1.0 = 2000 members, default: 0.5)",
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
        "-device",
        type=int,
        default=1,
        required=False,
        help="Device to use for LBE training (default: 1)",
    )
    args = parser.parse_args()

    experiment_lbe_with_prior(
        args.data,
        args.nsym,
        args.lbe_model,
        # args.unl_mem_ratio,
        args.results,
        p=args.prob,
        bins=args.bins,
        device=args.device,
    )


if __name__ == "__main__":
    main()
