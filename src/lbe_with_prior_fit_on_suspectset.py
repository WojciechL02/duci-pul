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


def prepare_data(name, seed, p, test_len=4000, run_type="real"):
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

    # Try to find files matching the patterns
    mem_matches1 = list(folder.glob(mem_pattern1))
    nonmem_matches1 = list(folder.glob(nonmem_pattern1))
    mem_matches2 = list(folder.glob(mem_pattern2))
    nonmem_matches2 = list(folder.glob(nonmem_pattern2))
    mem_matches3 = list(folder.glob(mem_pattern3))
    nonmem_matches3 = list(folder.glob(nonmem_pattern3))

    # Select patterns based on run_type
    if run_type == "real":
        # For real: Use pattern1 & pattern2 (files containing _real_*mem and _real_*nonmem)
        mem_matches = mem_matches1
        nonmem_matches = nonmem_matches1
        print(f"Run type: {run_type} - Using patterns 1 & 2 (real)")
    elif run_type == "synth":
        # For synth: Use pattern1 & pattern2 and pattern5 & pattern6 (files containing _real_*mem, _real_*nonmem, _from-mem, and _from-nonmem)
        mem_matches = mem_matches1 + mem_matches3
        nonmem_matches = nonmem_matches1 + nonmem_matches3
        print(f"Run type: {run_type} - Using patterns 1, 2, 5 & 6 (real and from-mem)")
    elif run_type == "ae_synth":
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
    elif run_type == "real":
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

    # shuffle with one permutation for all arrays
    # Generate a single permutation large enough for all arrays
    perm = np.random.permutation(max(len(X_mem), len(X_nonmem), len(X_mem_generated), len(X_nonmem_generated)))
    # Apply the same permutation to all arrays (using appropriate slices)
    X_mem = X_mem[perm[:len(X_mem)]]
    X_nonmem = X_nonmem[perm[:len(X_nonmem)]]
    X_mem_generated = X_mem_generated[perm[:len(X_mem_generated)]]
    X_nonmem_generated = X_nonmem_generated[perm[:len(X_nonmem_generated)]]
    # -----------------------------
    # Test set construction (unlabeled only)
    # -----------------------------
    n_test_unl = test_len - 2000
    n_pos_test = int(n_test_unl * p)  # members inside test unlabeled
    n_unl_test_nonmem = n_test_unl - n_pos_test  # non-members inside test unlabeled

    if run_type == "real":
        X_test_nonmem = np.concatenate(
            [
                X_mem_generated[:1000],
                X_nonmem_generated[1000:2000],
            ]
        )
    else:
        # generated data from examples in the suspect set
        X_test_nonmem = np.concatenate(
            [
                X_mem_generated[:n_pos_test],
                X_nonmem_generated[2000:2000 + n_unl_test_nonmem],
            ]
        )

    X_test_unl = np.concatenate(
        [
            X_test_nonmem,
            X_mem[:n_pos_test],
            X_nonmem[2000:2000+n_unl_test_nonmem],
        ]
    )

    y_test_unl = np.concatenate(
        [
            np.ones(2000, dtype=int), #NM_labeled (possibly generated in run_tymes synth)
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
    run_type="real",
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
            name=name, seed=sym, p=p, run_type=run_type
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
    parser.add_argument(
        "-run_type",
        type=str,
        default="real",
        required=False,
        choices=["real", "synth", "ae_synth"],
        help="Type of run to perform: real (pattern1&2), synth (pattern1&2 and pattern5&6), ae_synth (pattern3&4 and pattern5&6) (default: real)",
    )
    args = parser.parse_args()
    print(args.run_type)
    experiment_lbe_with_prior(
        args.data,
        args.nsym,
        args.lbe_model,
        # args.unl_mem_ratio,
        args.results,
        p=args.prob,
        bins=args.bins,
        device=args.device,
        run_type=args.run_type,
    )


if __name__ == "__main__":
    main()
