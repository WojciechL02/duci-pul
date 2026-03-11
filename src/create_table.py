import argparse
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

colors = {
    "real": "blue",
    "ae_synth": "red",
    "synth": "green",
    "correction": "gray",
}

labels = {
    "real": "Real",
    "ae_synth": "Synthetic (no correction)",
    "correction": "Synthetic (+correction) (ours)",
}

METHODS_MAPPING = {
    "km": "KM~\citep{ramaswamy2016km}",
    "tice": "TICE~\citep{bekker2018tice}",
    "dedpul": r"DEDPUL~\citep{ivanov2020dedpul}",
    "sumpe": r"SuMPE~\citep{zhu2023sumpe}",
    "alphamax": r"AlphaMax~\citep{jain2016alphamax}",
    "lbe": "PUL (LBE~\citep{gong2021lbe})",
    "threshold": "PUL (NTC-$\tau$MI~\citep{teser2025threshold})",
}

RAW_METHODS_ORDER = ["km", "tice", "dedpul", "sumpe", "alphamax", "pul"]

MODEL_DISPLAY = {
    "mar_b": "MAR-B",
    "mar_l": "MAR-L",
    "mar_h": "MAR-H",
    "rar_l": "RAR-L",
    "rar_xl": "RAR-XL",
    "rar_xxl": "RAR-XXL",
    "var_20": "VAR-20",
    "var_24": "VAR-24",
    "var_30": "VAR-30",
    "var_36": "VAR-36",
    "dit_rf": "DiT-RF-XL",
    "dit_rf_g": "DiT-RF-G",
    "uvit_t2i_deep": "UViT",
}


parser = argparse.ArgumentParser(
    description="Create p_hat_test comparison plots for a model directory."
)
parser.add_argument(
    "data_dir",
    type=str,
    help="Directory containing model results to plot",
)
parser.add_argument(
    "results_dir",
    type=str,
    help="Directory to save the result tables",
)
parser.add_argument(
    "--output_suffix", type=str, default="", help="Suffix to add to output filenames"
)
parser.add_argument(
    "--ss_len",
    type=int,
    nargs="+",
    default=[200, 500, 2000],
    help="Filter results by suspect set length(s) (e.g., 100 500 1000 1500)",
)
parser.add_argument(
    "--run_type",
    type=str,
    nargs="+",
    choices=["real", "ae_synth", "synth", "correction"],
    default=["real", "ae_synth", "synth", "correction"],
    help="Filter results by run type(s) (e.g., real ae_synth)",
)
parser.add_argument(
    "--targets",
    type=str,
    nargs="+",
    default=[
        "mar_b",
        "mar_l",
        "mar_h",
        "rar_l",
        "rar_xl",
        "rar_xxl",
        "var_20",
        "var_24",
        "var_30",
        "dit_rf",
        "uvit_t2i_deep",
    ],
    help="Filter results by model names (e.g., rar_xl rar_xxl var_24 var_30)",
)

args = parser.parse_args()


def parse_filename(filename: str) -> dict:
    KNOWN_RUN_TYPES = sorted(
        ["real", "synth", "ae_synth", "correction", "mean_mia_score", "tail"],
        key=len,
        reverse=True,
    )
    pattern = r"_len(\d+)(?:_(.*?))?_p=([\d\.]+)\.csv$"
    match = re.search(pattern, filename)

    if not match:
        raise ValueError(f"Filename format incorrect (tail missing): {filename}")

    ss_len, method_args, prob = match.groups()
    method_args = method_args if method_args else ""
    soup = filename[: match.start()]
    run_type = None
    for rt in KNOWN_RUN_TYPES:
        if soup.endswith(f"_{rt}"):
            run_type = rt
            soup = soup[: -(len(rt) + 1)]
            break

    if not run_type:
        raise ValueError(f"Unknown run_type in '{soup}'. Known: {KNOWN_RUN_TYPES}")

    method, _, target = soup.partition("_")
    return {
        "method": method,
        "target": target,
        "run_type": run_type,
        "ss_len": int(ss_len),
        "method_args": method_args,
        "prob": float(prob),
    }


def filter_files(files, ss_len=None, run_type=None, models=None):
    """
    Filter files based on specified criteria

    Args:
        files: List of files to filter
        ss_len: List of suspect set lengths to include
        run_type: List of run types to include
        models: List of model names to include

    Returns:
        List of files that match the criteria
    """
    filtered_files = []

    for file in files:
        file_data = parse_filename(file)

        if models is not None and file_data["target"] not in models:
            continue
        if ss_len is not None and file_data["ss_len"] not in ss_len:
            continue
        if run_type is not None and file_data["run_type"] not in run_type:
            continue
        filtered_files.append(file)
    return filtered_files


def load_records(
    results_dir: str,
    ss_len: list = None,
    run_type: list = None,
    models: list = None,
) -> dict:
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Directory {results_dir} does not exist.")

    all_files = [
        f for f in os.listdir(results_dir) if f.endswith(".csv") and "full" not in f
    ]
    if not all_files:
        raise ValueError(f"No CSV files found in {results_dir}.")

    matched_files = filter_files(
        all_files, ss_len=ss_len, run_type=run_type, models=models
    )
    if not matched_files:
        raise ValueError(f"No files match the specified criteria in {results_dir}.")

    records = defaultdict(lambda: defaultdict(list))

    for filename in matched_files:
        meta = parse_filename(filename)
        model = meta["target"]
        method_key = (
            f"{meta['method']}_{meta['method_args']}"
            if meta["method_args"]
            else meta["method"]
        )

        df = pd.read_csv(os.path.join(results_dir, filename), sep="\t")
        df["method"] = method_key
        df["p"] = meta["prob"]
        records[model][method_key].append(df)

    return {
        model: {
            method: pd.concat(dfs, ignore_index=True) for method, dfs in methods.items()
        }
        for model, methods in records.items()
    }


def create_paper_table(
    data_dir: str,
    results_dir: str,
    output_suffix: str = "",
    ss_len: list = None,
    run_type: list = None,
    models: list = None,
):
    records = load_records(data_dir, ss_len=ss_len, run_type=run_type, models=models)

    # Collect all methods across all models
    all_methods = set()
    for model_data in records.values():
        all_methods.update(model_data.keys())

    # Determine method order: prefer RAW_METHODS_ORDER, then alphabetical for the rest
    def method_sort_key(m):
        base = m.split("_")[0]
        if base in RAW_METHODS_ORDER:
            return (RAW_METHODS_ORDER.index(base), m)
        return (len(RAW_METHODS_ORDER), m)

    ordered_methods = sorted(all_methods, key=method_sort_key)
    ordered_models = (
        [m for m in models if m in records] if models else sorted(records.keys())
    )

    def compute_mae_metrics(df):
        """Compute mean MAE, std MAE, and max MAE across p values."""
        mae_per_p = df.groupby("p").apply(
            lambda g: (g["p_hat_test"] - g.name).abs().mean()
        )
        return mae_per_p.mean(), mae_per_p.std(), mae_per_p.max()

    # Build table data: rows=methods, columns=models
    table_data = {}
    for method in ordered_methods:
        table_data[method] = {}
        for model in ordered_models:
            if model in records and method in records[model]:
                df = records[model][method]
                mean_mae, std_mae, max_mae = compute_mae_metrics(df)
                table_data[method][model] = (mean_mae, std_mae, max_mae)
            else:
                table_data[method][model] = (None, None, None)

    # --- Build LaTeX ---
    n_models = len(ordered_models)

    # Column spec: method col + 2 cols per model
    col_spec = "l" + "".join(["cc"] * n_models)

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Top header row: model names spanning 2 columns each
    model_headers = " & ".join(
        r"\multicolumn{2}{c}{" + MODEL_DISPLAY.get(m, m) + "}" for m in ordered_models
    )
    lines.append(r"\textbf{Method} & " + model_headers + r" \\")

    # Cmidrules under each model name
    cmidrules = []
    for i, _ in enumerate(ordered_models):
        start = 2 + i * 2
        end = start + 1
        cmidrules.append(rf"\cmidrule(lr){{{start}-{end}}}")
    lines.append(" ".join(cmidrules))

    # Sub-header: MAE / Max MAE for each model
    sub_headers = " & ".join([r"\textbf{MAE} & \textbf{max MAE}"] * n_models)
    lines.append(r" & " + sub_headers + r" \\")
    lines.append(r"\midrule")

    # Data rows
    for method in ordered_methods:
        base_method = method.split("_")[0]
        method_label = METHODS_MAPPING.get(base_method, method)
        row_cells = [method_label]
        for model in ordered_models:
            mean_mae, std_mae, max_mae = table_data[method].get(
                model, (None, None, None)
            )
            if mean_mae is None:
                row_cells.extend(["--", "--"])
            else:
                mae_str = f"{mean_mae:.3f}{{\\tiny $\\pm${std_mae:.2f}}}"
                row_cells.extend([mae_str, f"{max_mae:.3f}"])
        lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{TODO}")
    lines.append(r"\end{table}")

    latex_str = "\n".join(lines)

    suffix = f"_{output_suffix}" if output_suffix else ""
    out_path = os.path.join(results_dir, f"paper_table{suffix}.tex")
    with open(out_path, "w") as f:
        f.write(latex_str)

    print(f"Saved LaTeX table to {out_path}")
    return latex_str


def main():
    data_dir = args.data_dir
    results_dir = args.results_dir
    output_suffix = args.output_suffix
    ss_len = args.ss_len
    run_type = args.run_type
    models = args.targets

    create_paper_table(
        data_dir,
        results_dir,
        output_suffix=output_suffix,
        ss_len=ss_len,
        run_type=run_type,
        models=models,
    )


if __name__ == "__main__":
    main()
