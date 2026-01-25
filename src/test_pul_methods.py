import argparse
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from data import prepare_data
from utils import seed_everything
from methods.pul import PUbasic, LBE, SAREM, PUThreshold, PUSB, PUe, PGlin, KM, TICE


METRICS_MAPPING = {
    "acc": r"\textbf{Accuracy}",
    "bacc": r"\textbf{Bal. Acc.}",
    "f1": r"\textbf{F1-Score}",
    "roc_auc": r"\textbf{AUC}",
    "time": r"\textbf{Time (s)}",
}

METHODS_MAPPING = {
    "oracle": "Oracle",
    "dummy": "Dummy",
    "threshold": "NTC-$\\tau$MI~\citep{teser2025threshold}",
    "sar-em": "SAR-EM~\citep{bekker2019sarem}",
    "pusb": "PUSB~\citep{kato2018pusb}",
    "pglin": "PGLIN~\citep{gerych2022recovering}",
    "lbe": "LBE~\citep{gong2021lbe}",
    "pue": "PUe~\citep{}",
}

METHODS_ORDER = [
    "Oracle",
    "Dummy",
    "Threshold",
    "LBE",
    "SAR-EM",
    "PGLIN",
    "PUe",
    "PUSB",
]

METRICS_ORDER = ["acc", "bacc", "f1", "roc_auc", "time"]


def save_paper_table(data: pd.DataFrame, name: str, path_to_plots: str) -> None:
    """
    Generates a single LaTeX table comparing PUL methods.
    """
    os.makedirs(path_to_plots, exist_ok=True)

    # 1. Aggregation
    available_metrics = [m for m in METRICS_ORDER if m in data.columns]
    grouped = data.groupby("method")[available_metrics]
    means = grouped.mean()
    stds = grouped.std()

    # 2. Formatting Values
    final_df = pd.DataFrame(index=means.index)

    for metric in available_metrics:
        final_df[METRICS_MAPPING[metric]] = means[metric].apply(
            lambda x: f"{x:.2f}"
        ) + stds[metric].apply(lambda x: f"{{\\tiny $\\pm${x:.2f}}}")

    # 3. Final Structure
    final_df.index = final_df.index.map(lambda x: METHODS_MAPPING.get(x, x))
    existing_methods = [m for m in METHODS_ORDER if m in final_df.index]
    other_methods = [m for m in final_df.index if m not in existing_methods]
    final_df = final_df.reindex(existing_methods + other_methods)

    # Reset index to make "Method" a proper column with a bold header
    final_df = final_df.reset_index()
    final_df.rename(columns={"method": r"\textbf{Method}"}, inplace=True)

    # 4. Export to LaTeX
    output_path = f"{path_to_plots}/pul_comparison_{name}.tex"
    col_format = "l" + "c" * (len(final_df.columns) - 1)

    final_df.to_latex(
        output_path,
        index=False,
        escape=False,
        column_format=col_format,
        header=True,
        caption=name,
    )


def compute_metrics(prob_y_pred, y_true):
    fpr_thr, tpr_thr, thr = roc_curve(y_true, prob_y_pred, pos_label=1)
    return {
        "acc": accuracy_score(y_true, np.where(prob_y_pred > 0.5, 1, 0)),
        "bacc": balanced_accuracy_score(y_true, np.where(prob_y_pred > 0.5, 1, 0)),
        "rec": recall_score(y_true, np.where(prob_y_pred > 0.5, 1, 0)),
        "prec": precision_score(y_true, np.where(prob_y_pred > 0.5, 1, 0)),
        "f1": f1_score(y_true, np.where(prob_y_pred > 0.5, 1, 0)),
        "roc_auc": auc(fpr_thr, tpr_thr),
    }


def run_experiment(methods, name, nsym, p, data_type, results_dir, data_dir):
    records = []
    for method in methods:
        for sym in np.arange(0, nsym, 1):
            X_data, y_data, s_data, _ = prepare_data(
                name=name,
                seed=int(sym),
                p=p,
                ss_len=2000,
                run_type=data_type,
                data_dir=data_dir,
            )
            X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
                X_data, y_data, s_data, test_size=0.2, random_state=int(sym)
            )
            seed_everything(int(sym))

            start_time = time.perf_counter()
            if method == "oracle":
                model = PUbasic()
                model.fit(X_train, y_train)
            elif method == "dummy":
                model = PUbasic()
                model.fit(X_train, s_train)
            elif method == "km":
                st = time.perf_counter()
                km_estimator = KM()
                est = km_estimator.estimate(X_train, s_train)
                print("TIME:", time.perf_counter() - st)
                km1 = 1 - est["alpha"]
                km2 = 1 - est["km2"]
                print(f"p={p}, km1={km1:.3f}, km2={km2:.3f}")
                break
            elif method == "tice":
                tice_estimator = TICE()
                est = tice_estimator.estimate(X_train, s_train)
                print(f"p={p}, p_hat={est['alpha']:.3f}")
            else:
                if method == "threshold":
                    model = PUThreshold()
                elif method == "sar-em":
                    model = SAREM()
                elif method == "pue":
                    model = PUe()
                elif method == "lbe":
                    model = LBE()
                elif method == "pglin":
                    model = PGlin()
                elif method == "pusb":
                    km_estimator = KM()
                    est = km_estimator.estimate(X_train, s_train)
                    est_pi = (1 - np.mean(s_train)) * est["km2"] + np.mean(s_train)
                    model = PUSB(est_pi, X_test, y_test)
                else:
                    raise ValueError(f"Method {method} not known.")
                model.fit(X_train, s_train)

    #         end_time = time.perf_counter()
    #         run_time = end_time - start_time
    #         prob_y_test = model.predict_proba(X_test)[:, 1]
    #         # flip_labels
    #         prob_y_test = 1 - prob_y_test
    #         y_test = 1 - y_test
    #
    #         results = compute_metrics(prob_y_test, y_test)
    #         results["time"] = run_time
    #         results["method"] = method
    #         records.append(results)
    #
    # df = pd.DataFrame(records)
    # save_paper_table(df, f"{name}_{p}", results_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=[
            # "pusb",
            # "threshold",
            # "sar-em",
            # "pglin",
            # "pue",
            # "lbe",
            # "oracle",
            # "dummy",
            "tice",
        ],
        required=False,
        help="PUL methods to compare",
    ),
    parser.add_argument("--target", type=str, required=True, help="Model to use.")
    parser.add_argument("--n_runs", type=int, required=True, help="Number of runs")
    parser.add_argument(
        "--prob", type=float, required=True, help="Probability value (between 0 and 1)"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="../results/ablations",
        required=False,
        help="Directory to save results (default: ../results/ablations)",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="real",
        required=False,
        choices=["real", "synth", "ae_synth"],
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/standard",
        required=False,
    )
    args = parser.parse_args()

    print(
        {
            "methods": args.methods,
            "target": args.target,
            "n_runs": args.n_runs,
            "data_type": args.data_type,
            "results_dir": args.results_dir,
            "data_dir": args.data_dir,
        }
    )
    # for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    p = args.prob
    print(f"p={p}")
    run_experiment(
        args.methods,
        args.target,
        args.n_runs,
        p,
        args.data_type,
        args.results_dir,
        args.data_dir,
    )
    print("Done!")


if __name__ == "__main__":
    main()
