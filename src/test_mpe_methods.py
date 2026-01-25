import argparse
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from data import prepare_data
from utils import seed_everything
from methods.pul import (
    KM,
    TICE,
    DEDPUL,
)

METHODS_MAPPING = {
    "km": "KM",
    "tice": "TICE",
    "dedpul": r"DEDPUL~\citep{}",
    "sumpe": r"SuMPE~\citep{}",
}

RAW_METHODS_ORDER = ["km", "tice", "dedpul", "sumpe"]


def save_paper_table(records: dict, name: str, path_to_plots: str) -> None:
    """
    Generates a LaTeX table from a records dictionary where:
    rows = methods, columns = p values, cell = mean +/- std.
    """
    os.makedirs(path_to_plots, exist_ok=True)
    p_values = sorted(records.keys())
    data = {}
    for p in p_values:
        method_dict = records[p]
        col_data = {}

        for method, scores in method_dict.items():
            if not scores:
                col_data[method] = "N/A"
                continue

            # Calculate Mean and Std
            mean_val = np.mean(scores)
            std_val = np.std(scores)

            # Format: 0.00 {\tiny \pm 0.00}
            formatted_cell = f"{mean_val:.2f}{{\\tiny $\\pm${std_val:.2f}}}"
            col_data[method] = formatted_cell

        col = f"$p={p}$"
        data[col] = col_data

    df = pd.DataFrame(data)
    existing_raw_methods = [m for m in RAW_METHODS_ORDER if m in df.index]
    other_raw_methods = [m for m in df.index if m not in existing_raw_methods]
    df = df.reindex(existing_raw_methods + other_raw_methods)
    df.index = df.index.map(lambda x: METHODS_MAPPING.get(x, x))
    df.index.name = r"\textbf{Method}"
    df = df.reset_index()
    output_path = f"{path_to_plots}/mpe_comparison_{name}.tex"
    col_format = "l" + "c" * (len(df.columns) - 1)
    df.to_latex(
        output_path,
        index=False,
        escape=False,
        column_format=col_format,
        header=True,
        caption=f"Comparison of MPE methods for {name}",
    )


def run_experiment(methods, name, nsym, p, ss_len, data_type, results_dir, data_dir):
    records = defaultdict(list)
    for method in methods:
        for sym in np.arange(0, nsym, 1):
            X_data, y_data, s_data, _ = prepare_data(
                name=name,
                seed=int(sym),
                p=p,
                ss_len=ss_len,
                run_type=data_type,
                data_dir=data_dir,
            )
            X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
                X_data, y_data, s_data, test_size=0.2, random_state=int(sym)
            )
            seed_everything(int(sym))

            if method == "km":
                km_estimator = KM()
                est = km_estimator.estimate(X_train, s_train)
                est["alpha"] = 1 - est["alpha"]
            elif method == "tice":
                tice_estimator = TICE()
                est = tice_estimator.estimate(X_train, s_train)
            elif method == "dedpul":
                dedpul_estimator = DEDPUL()
                est = dedpul_estimator.estimate(X_train, s_train)
            elif method == "sumpe":
                est = {"alpha": 0.0}
            else:
                raise ValueError(f"No known method {method}")

            records[method].append(est["alpha"])
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=[
            "km",
            "tice",
            "dedpul",
            "sumpe",
        ],
        required=False,
        help="MPE methods to compare",
    ),
    parser.add_argument("--target", type=str, required=True, help="Model to use.")
    parser.add_argument("--n_runs", type=int, required=True, help="Number of runs")
    parser.add_argument(
        "--ss_len", type=int, default=1000, required=False, help="Dataset size"
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
    records = {}
    for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        print(f"Running p={p}...")
        single_records = run_experiment(
            args.methods,
            args.target,
            args.n_runs,
            p,
            args.ss_len,
            args.data_type,
            args.results_dir,
            args.data_dir,
        )
        records[p] = single_records

    save_paper_table(records, f"{args.target}", args.results_dir)
    print("Done!")


if __name__ == "__main__":
    main()
