#!/usr/bin/env python3
import argparse
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import seaborn as sns

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "lines.linewidth": 2.2,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 1.0,
    "grid.color": "#cccccc",
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
})

# Base (darkest/saturated) color per run_type
RUN_TYPE_COLORS = {
    "real":       "#555555",
    "synth":      "#2ca02c",
    "ae_synth":   "#1f77b4",
    "correction": "#d62728",
}
RUN_TYPE_LABELS = {
    "real":       "Real",
    "synth":      "Synthetic",
    "ae_synth":   "Synthetic + AE",
    "correction": "Synthetic + AE + Corr.",
}
SIZE_ORDERS = {
    "rar": ["rar_l", "rar_xl", "rar_xxl"],
    "var": ["var_20", "var_24", "var_30"],
}

# Largest → solid; progressively smaller → more broken
SIZE_LINESTYLES = [":", "--", "-"]      # index 0 = smallest, -1 = largest
SIZE_MARKERS    = ["s", "^", "o"]      # index 0 = smallest, -1 = largest
SIZE_MARKERSIZES = [6, 7, 8]

# How much to lighten smaller sizes.
# 0.0 = base color, 1.0 = white.  Lightest first, darkest last.
SHADE_ALPHAS = [0.55, 0.30, 0.0]      # for 3 sizes; last one = full base color


def _shade_color(hex_color: str, blend: float) -> str:
    """Blend hex_color toward white by `blend` (0 = original, 1 = white)."""
    r, g, b = mcolors.to_rgb(hex_color)
    r2 = r + (1 - r) * blend
    g2 = g + (1 - g) * blend
    b2 = b + (1 - b) * blend
    return mcolors.to_hex((r2, g2, b2))


def parse_filename(filename: str) -> dict:
    KNOWN_RUN_TYPES = sorted(
        ["real", "synth", "ae_synth", "correction"],
        key=len, reverse=True
    )
    pattern = r"_len(\d+)(?:_(.*?))?_p=([\d\.]+)\.csv$"
    match = re.search(pattern, filename)
    if not match:
        return None
    ss_len, method_args, prob = match.groups()
    soup = filename[:match.start()]

    run_type = None
    for rt in KNOWN_RUN_TYPES:
        if soup.endswith(f"_{rt}"):
            run_type = rt
            soup = soup[: -(len(rt) + 1)]
            break

    if not run_type:
        return None
    method, _, target = soup.partition("_")
    family = "rar" if "rar" in target else ("var" if "var" in target else "other")

    return {
        "family": family,
        "dataset_size": target,
        "run_type": run_type,
        "ss_len": int(ss_len),
        "true_p": float(prob),
        "filename": filename,
    }


def build_dataframe(results_dir, ss_len_filter=None):
    print(f"Scanning {results_dir}...")
    records = []

    for file in os.listdir(results_dir):
        if not file.endswith(".csv") or "full" in file:
            continue
        meta = parse_filename(file)
        if not meta or meta["family"] == "other":
            continue
        if ss_len_filter and meta["ss_len"] not in ss_len_filter:
            continue

        file_path = os.path.join(results_dir, file)
        try:
            df = pd.read_csv(file_path, delimiter="\t")
            if meta["run_type"] != "correction" and "p_hat_test" in df.columns:
                pred_p = df["p_hat_test"].mean()
            elif meta["run_type"] == "correction" and "p_hat_2MIA-comb" in df.columns:
                pred_p = df["p_hat_2MIA-comb"].mean()
            else:
                continue
            meta["pred_p"] = pred_p
            meta["abs_error"] = abs(meta["true_p"] - pred_p)
            records.append(meta)
        except Exception:
            pass

    return pd.DataFrame(records)


def _draw_calibration_panel(ax, plot_df, sizes, run_type):
    """
    Draw one calibration panel.
    - Color = run_type base color, lightened for smaller sizes
    - Linestyle = dotted/dashed for small, solid for largest
    - Marker = varies per size
    """
    base_color = RUN_TYPE_COLORS.get(run_type, "steelblue")
    n = len(sizes)

    # Pad shade/style lists if fewer than 3 sizes
    shade_alphas  = SHADE_ALPHAS[-n:]
    linestyles    = SIZE_LINESTYLES[-n:]
    markers       = SIZE_MARKERS[-n:]
    markersizes   = SIZE_MARKERSIZES[-n:]

    for i, size in enumerate(sizes):
        size_df = plot_df[plot_df["dataset_size"] == size].sort_values("true_p")
        if size_df.empty:
            continue

        color = _shade_color(base_color, shade_alphas[i])

        ax.plot(
            size_df["true_p"],
            size_df["pred_p"],
            label=size,
            color=color,
            linestyle=linestyles[i],
            marker=markers[i],
            markersize=markersizes[i],
            linewidth=2.2,
            markeredgewidth=0.8,
            markeredgecolor="white",
            zorder=3,
        )

    # Ideal y = x
    ax.plot(
        [0, 1], [0, 1],
        linestyle="--", color="black", linewidth=2.0,
        alpha=0.75, label="Ideal", zorder=2,
    )

    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.set_xlabel("True p")
    ax.set_ylabel("Estimated p")
    ax.grid(True, zorder=0)

    ax.legend(framealpha=0.95, loc="upper left")


def plot_calibration_facets(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    run_types_present = df["run_type"].unique()

    for run_type in run_types_present:
        rt_df = df[df["run_type"] == run_type].copy()
        families_present = [f for f in SIZE_ORDERS if f in rt_df["family"].values]
        if not families_present:
            continue

        for family in families_present:
            sizes = SIZE_ORDERS[family]
            plot_df = (
                rt_df[rt_df["family"] == family]
                .groupby(["dataset_size", "true_p"])["pred_p"]
                .mean()
                .reset_index()
            )
            if plot_df.empty:
                continue

            fig, ax = plt.subplots(figsize=(5.5, 5.2))
            _draw_calibration_panel(ax, plot_df, sizes, run_type)
            fig.tight_layout(pad=1.5)

            for ext in ("png", "pdf"):
                out_path = os.path.join(output_dir, f"calibration_{run_type}_{family}.{ext}")
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
                print(f"Saved: {out_path}")

            plt.close(fig)


def plot_scaling_laws(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    mae_df = (
        df.groupby(["family", "dataset_size", "run_type"])["abs_error"]
        .mean()
        .reset_index()
        .rename(columns={"abs_error": "MAE"})
    )

    for family, sizes in SIZE_ORDERS.items():
        family_df = mae_df[mae_df["family"] == family]
        if family_df.empty:
            continue

        plt.figure(figsize=(7, 5))
        sns.pointplot(
            data=family_df,
            x="dataset_size",
            y="MAE",
            hue="run_type",
            order=sizes,
            palette=RUN_TYPE_COLORS,
            markers=["o", "s", "D", "^"],
        )
        plt.xlabel("Target Model Size")
        plt.ylabel("Mean Absolute Error")
        plt.grid(True, linestyle="--", alpha=0.6)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, [RUN_TYPE_LABELS.get(l, l) for l in labels], title="Run Type")

        out_path = os.path.join(output_dir, f"{family}_scaling_law.png")
        out_path1 = os.path.join(output_dir, f"{family}_scaling_law.pdf")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.savefig(out_path1, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    parser.add_argument("--ss_len", type=int, nargs="+", default=[2000])
    args = parser.parse_args()

    df = build_dataframe(args.results_dir, ss_len_filter=args.ss_len)
    if df.empty:
        print("No valid data found.")
        exit()

    output_dir = os.path.join(args.results_dir, "scaling_plots")
    plot_calibration_facets(df, output_dir)
    plot_scaling_laws(df, output_dir)