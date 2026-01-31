import argparse
import os
import time
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from data import prepare_data
from utils import seed_everything
from methods.algorithms import (
    KM,
    TICE,
    DEDPUL,
    SuMPE,
    AlphaMax,
)
from methods.pul_based import PULBased
import concurrent.futures


plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "font.size": 15,  # Set font size to 11pt
        "axes.labelsize": 15,  # -> axis labels
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 2,
        "text.usetex": False,
        "pgf.rcfonts": False,
    }
)

METHODS_MAPPING = {
    "km": "KM~\citep{ramaswamy2016km}",
    "tice": "TICE~\citep{bekker2018tice}",
    "dedpul": r"DEDPUL~\citep{ivanov2020dedpul}",
    "sumpe": r"SuMPE~\citep{zhu2023sumpe}",
    "alphamax": r"AlphaMax~\citep{jain2016alphamax}",
    "pul": "PUL",
}

COLOR_MAPPING = {
    "km": "red",
    "tice": "green",
    "dedpul": "blue",
    "sumpe": "orange",
    "alphamax": "purple",
    "pul": "gray",
}

RAW_METHODS_ORDER = ["km", "tice", "dedpul", "sumpe", "alphamax", "pul"]


def calculate_metrics_df(records: dict) -> pd.DataFrame:
    """
    Transforms raw records into a DataFrame with MAE, RMSE, and RE for all p.
    """
    rows = []
    p_values = sorted(records.keys())

    for p in p_values:
        method_dict = records[p]
        for method, scores in method_dict.items():
            if not scores:
                continue

            y_pred = np.array(scores)
            y_true = p

            # 1. MAE (Mean Absolute Error)
            mae = np.mean(np.abs(y_pred - y_true))

            # 2. RMSE (Root Mean Squared Error)
            rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

            # 3. RE (Relative Error) - Handle division by zero
            if y_true > 0:
                re = np.mean(np.abs(y_pred - y_true)) / y_true
            else:
                re = np.nan

            rows.append({"p": p, "method": method, "MAE": mae, "RMSE": rmse, "RE": re})

    return pd.DataFrame(rows)


def save_prediction_plot(records: dict, path_to_plots: str, name: str):
    """
    Generates a Calibration Plot:
    - X-axis: Real p
    - Y-axis: Mean Predicted p
    - Shaded Region: Mean +/- Std Dev (Light Gray)
    """
    os.makedirs(path_to_plots, exist_ok=True)
    plot_data = {}
    all_p_values = sorted(records.keys())

    all_methods = set()
    for p in all_p_values:
        all_methods.update(records[p].keys())

    for method in all_methods:
        real_p = []
        mean_pred = []
        std_pred = []

        for p in all_p_values:
            if method in records[p] and records[p][method]:
                scores = records[p][method]
                real_p.append(p)
                mean_pred.append(np.mean(scores))
                std_pred.append(np.std(scores))

        if real_p:
            plot_data[method] = {
                "x": np.array(real_p),
                "y": np.array(mean_pred),
                "std": np.array(std_pred),
            }

    plt.figure(figsize=(8, 8))
    # Plot Ideal Diagonal Line
    plt.plot(
        [0, 1],
        [0, 1],
        color="black",
        linestyle="--",
        label="Ideal",
        zorder=0,
    )

    for i, method in enumerate(sorted(plot_data.keys())):
        data = plot_data[method]
        method_label = (METHODS_MAPPING.get(method, method)).split("~")[0]
        color = COLOR_MAPPING[method]
        # Plot Mean Line
        plt.plot(
            data["x"],
            data["y"],
            label=method_label,
            color=color,
            marker="o",
            markersize=4,
        )
        # Plot Std Deviation (Light Gray Shading)
        lower_bound = data["y"] - data["std"]
        upper_bound = data["y"] + data["std"]

        plt.fill_between(
            data["x"],
            lower_bound,
            upper_bound,
            color=color,
            alpha=0.1,
            label="_nolegend_",
        )

    n = name.replace("_", "-").upper()
    plt.title(f"Estimation Performance: {n}")
    plt.ylabel("Predicted Proportion ($\hat{p}$)")
    plt.xlabel("Real Proportion ($p$)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Method", loc="upper left")
    plt.tight_layout()
    filename = f"{path_to_plots}/plot_calibration_{name}.png"
    plt.savefig(filename, dpi=300)
    plt.close()


def save_paper_table(records: dict, name: str, path_to_plots: str) -> None:
    """
    Generates the LaTeX Table (MAE/RMSE/RE) and calls the Plotting function.
    """
    # 1. Generate the Plot (Predicted vs Real)
    save_prediction_plot(records, path_to_plots, name)

    # 2. Prepare Data for the Table
    df_metrics = calculate_metrics_df(records)

    # 3. Configure Table Columns
    # We want specific p values with specific metrics
    target_configs = [
        (0.05, "RE", "RE"),  # (p_value, metric_key, column_header)
        (0.2, "MAE", "MAE"),  # Change "MAE" to "RMSE" here if preferred
        (0.5, "MAE", "MAE"),
        (0.8, "MAE", "MAE"),
        (0.2, "RMSE", "RMSE"),
        (0.5, "RMSE", "RMSE"),
        (0.8, "RMSE", "RMSE"),
    ]

    data_for_table = {}

    methods = df_metrics["method"].unique()
    known = [m for m in RAW_METHODS_ORDER if m in methods]
    unknown = [m for m in methods if m not in known]
    methods = known + unknown

    col_format = "l"
    prev_metric_group = None

    for p_val, metric, metric_name in target_configs:
        if prev_metric_group is None:
            col_format += "|c"
        elif metric_name != prev_metric_group:
            col_format += "|c"
        else:
            col_format += "c"

        prev_metric_group = metric_name

        subset = df_metrics[df_metrics["p"] == p_val].set_index("method")
        best_score = subset[metric].min() if not subset.empty else 0

        col_values = []
        for m in methods:
            if m in subset.index:
                val = subset.loc[m, metric]
                is_best = abs(val - best_score) < 1e-9
                fmt_val = f"\\textbf{{{val:.3f}}}" if is_best else f"{val:.3f}"
                col_values.append(fmt_val)
            else:
                col_values.append("N/A")

        # Stacked Header using standard LaTeX tabular
        header = f"\\begin{{tabular}}{{@{{}}c@{{}}}}{metric_name} \\\\ {{\\small $p={p_val}$}}\\end{{tabular}}"
        data_for_table[header] = col_values

    # 4. Create DataFrame
    df_latex = pd.DataFrame(data_for_table, index=methods)
    if "METHODS_MAPPING" in globals():
        df_latex.index = df_latex.index.map(lambda x: METHODS_MAPPING.get(x, x))

    df_latex.index.name = r"\textbf{Method}"
    df_latex = df_latex.reset_index()
    output_path = f"{path_to_plots}/mpe_results_{name}.tex"

    n = name.replace("_", "-").upper()
    df_latex.to_latex(
        output_path,
        index=False,
        escape=False,
        column_format=col_format,
        header=True,
        caption=f"MPE Performance for {n}.",
    )


def run_experiment(methods, name, nsym, p, ss_len, data_type, data_dir):
    records = defaultdict(list)
    for method in methods:
        for sym in range(nsym):
            X_data, y_data, s_data, _ = prepare_data(
                name=name,
                seed=sym,
                p=p,
                ss_len=ss_len,
                run_type=data_type,
                data_dir=data_dir,
            )
            seed_everything(sym)

            if method == "km":
                km_estimator = KM(stability_eps=1e-9)
                est = km_estimator.estimate(X_data, s_data)
                est["alpha"] = 1 - est["alpha"]
            elif method == "tice":
                tice_estimator = TICE()
                est = tice_estimator.estimate(X_data, s_data)
            elif method == "dedpul":
                dedpul_estimator = DEDPUL()
                est = dedpul_estimator.estimate(X_data, s_data)
            elif method == "sumpe":
                estimator = SuMPE(
                    base_estimator="km", base_estimator_kwargs={"stability_eps": 1e-9}
                )
                est = estimator.estimate(X_data, s_data)
            elif method == "alphamax":
                estimator = AlphaMax()
                est = estimator.estimate(X_data, s_data)
                est["alpha"] = 1 - est["alpha"]
            elif method == "pul":
                estimator = PULBased(
                    "lbe", {"kind": "MLP"}, {"bins": 5}, seed=sym, run_type=data_type
                )
                est = estimator.fit_estimate(X_data, y_data, s_data)
                est["alpha"] = est["p_hat_test"]
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
        default=["km", "tice", "dedpul", "sumpe", "alphamax", "pul"],
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
    # records = {}
    # for p in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    #     print(f"Running p={p}...")
    #     st = time.perf_counter()
    #     single_records = run_experiment(
    #         args.methods,
    #         args.target,
    #         args.n_runs,
    #         p,
    #         args.ss_len,
    #         args.data_type,
    #         args.data_dir,
    #     )
    #     records[p] = single_records
    #     et = time.perf_counter()
    #     print(f"Time: {et-st:.1f}(s)")
    #
    # save_paper_table(records, f"{args.target}", args.results_dir)
    # print("Done!")

    p_values = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    records = {}
    print(f"Starting parallel execution...")
    global_start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_p = {
            executor.submit(
                run_experiment,
                args.methods,
                args.target,
                args.n_runs,
                p,
                args.ss_len,
                args.data_type,
                args.data_dir,
            ): p
            for p in p_values
        }

        for future in concurrent.futures.as_completed(future_to_p):
            p = future_to_p[future]
            try:
                single_records = future.result()
                records[p] = single_records
                print(f"Finished p={p}")
            except Exception as exc:
                print(f"p={p} generated an exception: {exc}")
    global_end = time.perf_counter()
    print(f"Total time: {global_end - global_start:.1f}(s)")

    save_paper_table(records, f"{args.target}", args.results_dir)
    print("Done!")


if __name__ == "__main__":
    main()
