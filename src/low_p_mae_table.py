#!/usr/bin/env python3
"""Build a single, paper-ready MAE table from results/neurips/low_p CSVs.

Produces ONE wide LaTeX table for the low-p regime -- p values are COLUMNS
(0.05, 0.10, ..., 0.50), so the per-p trend is preserved instead of being
averaged away. Each target model gets a 4-row group: Real, Synth, Synth+AE,
and a highlighted Delta row (MAE_Synth - MAE_Synth+AE, positive = AE better)
so the advantage of Synth+AE over raw Synth at low p is visible at a glance,
without needing 10 separate tables.

A plain-text/CSV summary per p is also written (not LaTeX) for inspection,
plus a delta-MAE (synth - ae_synth) CSV across all p values.
"""
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

RESULTS_DIR = "results/neurips/low_p"
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
RUN_TYPES = ["real", "synth", "ae_synth"]
P_VALUES = [round(0.05 * i, 2) for i in range(1, 11)]  # 0.05 .. 0.50

RUN_TYPE_LABELS = {"real": "Real", "synth": "Synth", "ae_synth": "Synth + AE"}

METHOD_LABELS = {
    "tice": r"TIcE~\citep{bekker2018tice}",
    "alphamax": r"AlphaMax~\citep{jain2016alphamax}",
    "lbe": r"PUL (LBE~\citep{gong2021lbe})",
    "threshold": r"PUL (NTC-$\tau$MI~\citep{teser2025threshold})",
}

TARGET_LABELS = {
    "rar_xl": "RAR-XL",
    "rar_xxl": "RAR-XXL",
    "var_24": "VAR-24",
    "var_30": "VAR-30",
    "var_36": "VAR-36",
    "dit_rf": "DiT-RF-XL",
    "dit_rf_b": "DiT-RF-B",
    "dit_rf_g": "DiT-RF-G",
    "mar_b": "MAR-B",
    "mar_l": "MAR-L",
    "mar_h": "MAR-H",
    "uvit_t2i_deep": "U-ViT-T2I-Deep",
    "minfm": "MinFM",
}

# {method}_{target}_{real|synth|ae_synth}_len{ss_len}_..._p={p}.csv
FILENAME_RE = re.compile(
    r"^(?P<method>[a-zA-Z]+)_(?P<target>.+?)_(?P<run_type>real|synth|ae_synth)_"
    r"len(?P<ss_len>\d+).*_p=(?P<p>[\d.]+)\.csv$"
)


def parse_filename(fname):
    m = FILENAME_RE.match(fname)
    if not m:
        return None
    d = m.groupdict()
    d["ss_len"] = int(d["ss_len"])
    d["p"] = float(d["p"])
    return d


def method_label(method):
    return METHOD_LABELS.get(method, method.upper())


def target_label(target):
    return TARGET_LABELS.get(target, target.replace("_", "-").upper())


# ---------------------------------------------------------------------------
# 1. Load every csv, compute MAE mean/std per (method, target, run_type, p)
# ---------------------------------------------------------------------------
# records[p][run_type][method][target] = {"mean": ..., "std": ..., "raw": [errors]}
records = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
ss_lens = set()

for fname in os.listdir(RESULTS_DIR):
    if not fname.endswith(".csv"):
        continue
    meta = parse_filename(fname)
    if meta is None:
        continue
    if meta["p"] not in P_VALUES:
        continue

    df = pd.read_csv(os.path.join(RESULTS_DIR, fname), sep="\t")
    p_hat = df["p_hat_test"]
    mae = (p_hat - meta["p"]).abs()

    records[meta["p"]][meta["run_type"]][meta["method"]][meta["target"]] = {
        "mean": mae.mean(),
        "std": mae.std(),
        "raw": mae.to_numpy(),
    }
    ss_lens.add(meta["ss_len"])

if not records:
    raise SystemExit(f"No matching result files found in {RESULTS_DIR}.")

ss_len = ss_lens.pop() if len(ss_lens) == 1 else "/".join(map(str, sorted(ss_lens)))

# All methods/targets observed anywhere, kept in a stable, sorted order.
all_methods = sorted({m for p in records for rt in records[p] for m in records[p][rt]})
all_targets = sorted({t for p in records for rt in records[p] for m in records[p][rt]
                       for t in records[p][rt][m]})


# ---------------------------------------------------------------------------
# 2. ONE wide LaTeX table: p values as COLUMNS (so the per-p trend is kept,
#    not averaged away), one row-group per target, with a highlighted Delta
#    row (Synth - Synth+AE) that makes the AE advantage visible at a glance.
# ---------------------------------------------------------------------------
def method_for_target(target):
    """The (single, in this dataset) method that was run for this target."""
    for p in P_VALUES:
        for rt in RUN_TYPES:
            for m in records.get(p, {}).get(rt, {}):
                if target in records[p][rt][m]:
                    return m
    return None


def cell(rt, method, target, p):
    stats = records.get(p, {}).get(rt, {}).get(method, {}).get(target)
    if stats is None:
        return "--"
    return f"{stats['mean']:.3f}{{\\tiny $\\pm${stats['std']:.2f}}}"


def delta_cell(method, target, p):
    synth = records.get(p, {}).get("synth", {}).get(method, {}).get(target)
    ae = records.get(p, {}).get("ae_synth", {}).get(method, {}).get(target)
    if synth is None or ae is None:
        return "--"
    return f"{synth['mean'] - ae['mean']:+.3f}"


def build_wide_table():
    col_format = "@{}c@{}|l" + "c" * len(P_VALUES)

    lines = []
    lines.append(r"\begin{table*}[b!]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(
        r"\caption{\textbf{Dataset usage estimation in the low-prevalence regime} "
        f"(suspect set size {ss_len} images). For each target model and ground-truth "
        r"$p \in \{" + ", ".join(f"{p:g}" for p in P_VALUES) + r"\}$, we report the MAE "
        r"(mean {\tiny $\pm$std} over runs) of the estimated member ratio $\hat p$ under "
        r"the Real, Synth, and Synth~+~AE reference conditions. The $\Delta$ row is "
        r"$\text{MAE}_{\text{Synth}} - \text{MAE}_{\text{Synth+AE}}$: "
        r"\textbf{positive values indicate Synth~+~AE outperforms raw Synth}. Lower MAE "
        r"is better.}"
    )
    lines.append(r"\label{tab:low_p_per_p}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{col_format}}}")
    lines.append(r"\toprule")

    header = ["", ""] + [rf"$p={p:g}$" for p in P_VALUES]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    for t_idx, target in enumerate(all_targets):
        method = method_for_target(target)
        group_label = f"{target_label(target)}"
        if method is not None:
            group_label += rf" \tiny({method_label(method)})"

        lines.append(
            rf"\multirow{{4}}{{*}}{{\rotatebox{{90}}{{{group_label}}}}} & "
            + "Real & "
            + " & ".join(cell("real", method, target, p) for p in P_VALUES)
            + r" \\"
        )
        lines.append(
            " & Synth & " + " & ".join(cell("synth", method, target, p) for p in P_VALUES) + r" \\"
        )
        lines.append(
            r" & Synth + AE & "
            + " & ".join(cell("ae_synth", method, target, p) for p in P_VALUES)
            + r" \\"
        )
        lines.append(
            r" & \textbf{$\Delta$ (Synth$-$AE)} & \textbf{"
            + "} & \\textbf{".join(delta_cell(method, target, p) for p in P_VALUES)
            + r"} \\"
        )
        if t_idx != len(all_targets) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


os.makedirs(TABLES_DIR, exist_ok=True)
wide_tex = build_wide_table()
wide_path = os.path.join(TABLES_DIR, "table_low_p_per_p.tex")
with open(wide_path, "w", encoding="utf-8") as f:
    f.write(wide_tex + "\n")

print(f"Wrote 1 wide LaTeX table (p={P_VALUES} as columns) to {wide_path}\n")
print(wide_tex)
print()

# Plain-text/CSV echo per p, for quick sanity checking outside LaTeX.
for p in P_VALUES:
    rows = []
    for rt in RUN_TYPES:
        rt_data = records.get(p, {}).get(rt, {})
        for method in all_methods:
            for target in all_targets:
                stats = rt_data.get(method, {}).get(target)
                if stats is None:
                    continue
                rows.append(
                    {
                        "data": RUN_TYPE_LABELS[rt],
                        "method": method,
                        "target": target,
                        "MAE": f"{stats['mean']:.4f} ± {stats['std']:.4f}",
                    }
                )
    if not rows:
        continue
    df_p = pd.DataFrame(rows)
    print(f"--- p={p:g} ---")
    print(df_p.to_string(index=False))
    print()
    df_p.to_csv(os.path.join(TABLES_DIR, f"mae_p={p:g}.csv"), index=False)


# ---------------------------------------------------------------------------
# 3. Delta MAE (synth - ae_synth) across all p, one column per p (unchanged)
# ---------------------------------------------------------------------------
delta_rows = []
for method in all_methods:
    for target in all_targets:
        row = {"method": method, "target": target}
        for p in P_VALUES:
            synth = records.get(p, {}).get("synth", {}).get(method, {}).get(target)
            ae_synth = records.get(p, {}).get("ae_synth", {}).get(method, {}).get(target)
            if synth is None or ae_synth is None:
                row[f"p={p:g}"] = ""
            else:
                row[f"p={p:g}"] = f"{synth['mean'] - ae_synth['mean']:+.4f}"
        delta_rows.append(row)

table2 = pd.DataFrame(delta_rows).set_index(["method", "target"])
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
print("Delta MAE (synth - ae_synth), positive => ae_synth better\n")
print(table2.to_string())
table2.to_csv(os.path.join(RESULTS_DIR, "mae_delta_table.csv"))
