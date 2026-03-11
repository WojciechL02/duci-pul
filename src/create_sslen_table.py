import argparse
import os
import re
import pandas as pd
from collections import defaultdict


METHODS_MAPPING = {
    "km": "KM~\\citep{ramaswamy2016km}",
    "tice": "TICE~\\citep{bekker2018tice}",
    "dedpul": r"DEDPUL~\citep{ivanov2020dedpul}",
    "sumpe": r"SuMPE~\citep{zhu2023sumpe}",
    "alphamax": r"AlphaMax~\citep{jain2016alphamax}",
    "lbe": "PUL (LBE~\citep{gong2021lbe})",
    "threshold": "PUL (NTC-$\tau$MI~\citep{teser2025threshold})",
}

MODEL_DISPLAY = {
    "mar_b": "MAR-B", "mar_l": "MAR-L", "mar_h": "MAR-H",
    "rar_l": "RAR-L", "rar_xl": "RAR-XL", "rar_xxl": "RAR-XXL",
    "var_20": "VAR-20", "var_24": "VAR-24", "var_30": "VAR-30", "var_36": "VAR-36",
    "dit_rf": "DiT", "uvit_t2i_deep": "UViT",
}

parser = argparse.ArgumentParser(
    description="Create per-method ss_len comparison tables."
)
parser.add_argument("data_dir", type=str, help="Directory containing model results")
parser.add_argument("results_dir", type=str, help="Directory to save result tables")
parser.add_argument("--output_suffix", type=str, default="")
parser.add_argument(
    "--ss_len", type=int, nargs="+", default=[100, 500, 1000, 2000],
    help="Suspect set sizes to include as rows",
)
parser.add_argument(
    "--run_type", type=str, nargs="+",
    choices=["real", "ae_synth", "synth", "correction"],
    default=["real", "ae_synth", "synth", "correction"],
)
parser.add_argument(
    "--targets", type=str, nargs="+",
    default=["rar_xl", "rar_xxl", "var_24", "var_30", "dit_rf"],
)

args = parser.parse_args()


def parse_filename(filename: str) -> dict:
    KNOWN_RUN_TYPES = sorted(
        ["real", "synth", "ae_synth", "correction", "mean_mia_score", "tail"],
        key=len, reverse=True,
    )
    pattern = r"_len(\d+)(?:_(.*?))?_p=([\d\.]+)\.csv$"
    match = re.search(pattern, filename)
    if not match:
        raise ValueError(f"Filename format incorrect: {filename}")

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
        raise ValueError(f"Unknown run_type in '{soup}'")

    method, _, target = soup.partition("_")
    return {
        "method": method,
        "target": target,
        "run_type": run_type,
        "ss_len": int(ss_len),
        "method_args": method_args,
        "prob": float(prob),
    }


def load_records(data_dir, ss_len=None, run_type=None, models=None):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} does not exist.")

    all_files = [f for f in os.listdir(data_dir) if f.endswith(".csv") and "full" not in f]
    if not all_files:
        raise ValueError(f"No CSV files found in {data_dir}.")

    # Filter files
    matched = []
    for f in all_files:
        try:
            meta = parse_filename(f)
        except ValueError:
            continue
        if models is not None and meta["target"] not in models:
            continue
        if ss_len is not None and meta["ss_len"] not in ss_len:
            continue
        if run_type is not None and meta["run_type"] not in run_type:
            continue
        matched.append(f)

    if not matched:
        raise ValueError("No files match the specified criteria.")

    # records[model][method_key][ss_len] -> list of dfs
    records = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for filename in matched:
        meta = parse_filename(filename)
        model = meta["target"]
        method_key = f"{meta['method']}_{meta['method_args']}" if meta["method_args"] else meta["method"]
        df = pd.read_csv(os.path.join(data_dir, filename), sep="\t")
        df["p"] = meta["prob"]
        records[model][method_key][meta["ss_len"]].append(df)

    # Concat dfs
    result = {}
    for model, methods in records.items():
        result[model] = {}
        for method, lens in methods.items():
            result[model][method] = {
                sl: pd.concat(dfs, ignore_index=True)
                for sl, dfs in lens.items()
            }
    return result


def compute_mae_metrics(df):
    mae_per_p = df.groupby("p").apply(
        lambda g: (g["p_hat_test"] - g.name).abs().mean()
    )
    return mae_per_p.mean(), mae_per_p.std()


def build_table_for_method(method_key, records, ordered_models, ss_lens, caption=None, label=None):
    base_method = method_key.split("_")[0]
    method_label = METHODS_MAPPING.get(base_method, method_key)

    n_models = len(ordered_models)
    col_spec = "l" + "c" * n_models

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row
    model_headers = " & ".join(MODEL_DISPLAY.get(m, m) for m in ordered_models)
    lines.append(r"\textbf{Suspect set size} & " + model_headers + r" \\")
    lines.append(r"\midrule")

    # Data rows — one per ss_len
    for sl in ss_lens:
        row_cells = [str(sl)]
        for model in ordered_models:
            if model in records and method_key in records[model] and sl in records[model][method_key]:
                df = records[model][method_key][sl]
                mean_mae, std_mae = compute_mae_metrics(df)
                std_str = f"{std_mae:.2f}" if std_mae is not None else "---"
                cell = f"{mean_mae:.3f}{{\\tiny$\\pm${std_str}}}"
            else:
                cell = "--"
            row_cells.append(cell)
        lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    cap = caption or (
        f"\\textbf{{MAE {method_label}.}} "
        r"Lower is better."
    )
    lbl = label or f"tab:mae_{method_key}"
    lines.append(f"\\caption{{{cap}}}")
    lines.append(f"\\label{{{lbl}}}")
    lines.append(r"\vspace{-0.4cm}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def create_ss_len_tables(
    data_dir, results_dir, output_suffix="", ss_len=None, run_type=None, models=None
):
    os.makedirs(results_dir, exist_ok=True)
    records = load_records(data_dir, ss_len=ss_len, run_type=run_type, models=models)

    # Collect all method keys
    all_methods = set()
    for model_data in records.values():
        all_methods.update(model_data.keys())

    ordered_models = [m for m in models if m in records] if models else sorted(records.keys())
    ss_lens_sorted = sorted(ss_len) if ss_len else sorted(
        {sl for m in records.values() for meth in m.values() for sl in meth.keys()}
    )

    suffix = f"_{output_suffix}" if output_suffix else ""

    for method_key in sorted(all_methods):
        latex = build_table_for_method(method_key, records, ordered_models, ss_lens_sorted)
        out_path = os.path.join(results_dir, f"ss_len_table_{method_key}{suffix}.tex")
        with open(out_path, "w") as f:
            f.write(latex)
        print(f"Saved: {out_path}")


def main():
    create_ss_len_tables(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        output_suffix=args.output_suffix,
        ss_len=args.ss_len,
        run_type=args.run_type,
        models=args.targets,
    )


if __name__ == "__main__":
    main()
