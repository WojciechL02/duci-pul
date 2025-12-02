import argparse
import time
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from data import prepare_data
from utils import no_positives_test, save_results, print_summary
from PUBiasCalibration.Models.LBEWithPrior import (
    LBEWithPrior,
    seed,
)


def parse_args():
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
        choices=[
            "mar_b",
            "mar_h",
            "mar_l",
            "rar_b",
            "rar_l",
            "rar_xl",
            "rar_xxl",
            "var_16",
            "var_20",
            "var_24",
            "var_30",
            "dit_rf_all",
            "dit_rf_clid",
            "uvit_t2i_deep_clid",
            "pythia-6_9b",
            "pythia-12b",
        ],
        help="Model to use.",
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
        "-min_bin_count_N",
        type=int,
        default=5,
        required=False,
        help="Minimum number of bins in the Negative subset in p estimation (default: 5)",
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
    return parser.parse_args()


def main():
    args = parse_args()

    config = {
        "name": args.data,
        "nsym": args.nsym,
        "lbe_model": args.lbe_model,
        "results_dir": args.results,
        "ss_len": args.ss_len,
        "p": args.prob,
        "bins": args.bins,
        "device": args.device,
        "run_type": args.run_type,
    }

    metrics = [
        "acc",
        "bacc",
        "p_hat_test",
        "H0_p_value",
    ]
    if config["run_type"] == "correction":
        metrics += ["p_hat_ctrl", "p_hat_comb", "p_hat_2MIA-comb", "p_hat_MIA-ctrl"]

    print(80 * "=")
    print(f"Method={args.run_type} | Data={args.data} | p={args.prob}")
    print(80 * "=")
    print("\nConfig:")
    print(config)
    print(80 * "-")

    tstart = time.time()
    records = []
    for sym in np.arange(0, config["nsym"], 1):
        print(f"Run {sym+1}/{config['nsym']}")
        X_test, y_test, s_test, X_ctrl_test = prepare_data(
            name=args.data,
            seed=sym,
            p=args.prob,
            run_type=args.run_type,
            ss_len=args.ss_len,
        )
        np.random.seed(sym)
        seed(sym)

        start_time = time.time()
        model = LBEWithPrior(
            kind=args.lbe_model,
            bins=args.bins,
            min_bin_count_N=args.min_bin_count_N,
            device=args.device,
        )
        model.fit(X_test, s_test)
        end_time = time.time()
        run_time = end_time - start_time

        internal_pi = model.get_prior()

        if args.run_type == "correction" and X_ctrl_test is not None:
            X_comb_test = np.hstack([X_ctrl_test, X_test])

            start_time = time.time()
            mdl_ctrl = LBEWithPrior(
                kind=args.lbe_model,
                bins=args.bins,
                min_bin_count_N=args.min_bin_count_N,
                device=args.device,
            )
            mdl_ctrl.fit(X_ctrl_test, s_test)

            mdl_comb = LBEWithPrior(
                kind=args.lbe_model,
                bins=args.bins,
                min_bin_count_N=args.min_bin_count_N,
                device=args.device,
            )
            mdl_comb.fit(X_comb_test, s_test)
            end_time = time.time()
            run_time = end_time - start_time

            mdl_comb._fit_runtime = run_time  # stash for logging

            internal_pi_ctrl = mdl_ctrl.get_prior()
            internal_pi_comb = mdl_comb.get_prior()

        prob_y_test = model.predict_proba(X_test)[:, 1]

        p_value = (
            no_positives_test(
                prob_y_test[s_test == 0], prob_y_test[s_test == 1], delta=0.01
            )
        )["p_value"]

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

        if args.run_type == "correction" and X_ctrl_test is not None:
            results.update(
                {
                    "p_hat_ctrl": internal_pi_ctrl,
                    "p_hat_comb": internal_pi_comb,
                    "p_hat_2MIA-comb": 2 * internal_pi - internal_pi_comb,
                    "p_hat_MIA-ctrl": internal_pi - internal_pi_ctrl,
                    "H0_p_value": p_value,
                }
            )
        records.append(results)

    df = save_results(records, metrics, args.results, config)
    print_summary(df, config)
    print("[Elapsed time = {:.1f} min]".format((time.time() - tstart) / 60))
    print("Done!")


if __name__ == "__main__":
    main()
