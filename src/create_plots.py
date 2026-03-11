#!/usr/bin/env python3
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
    "real": "gray",
    "ae_synth": "blue",
    "synth": "green",
    "correction": "red",
}

labels = {
    "real": "Real",
    "synth": "Synthetic",
    "ae_synth": "Synthetic + AE",
    "correction": "Synthetic + AE + Corr.",
}


parser = argparse.ArgumentParser(
    description="Create p_hat_test comparison plots for a model directory."
)
parser.add_argument(
    "results_dir",
    type=str,
    help="Directory containing model results to plot",
)
parser.add_argument(
    "--output_suffix", type=str, default="", help="Suffix to add to output filenames"
)
parser.add_argument(
    "--no-std",
    dest="show_std",
    action="store_false",
    help="Do not show standard deviation in plots (default: show std)",
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
        "dit_rf_g",
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


def create_individual_model_plots(
    results_dir,
    target,
    files_grouped,
    output_suffix="",
    show_std=True,
    ss_len=None,
    run_type=None,
    models=None,
):
    """
    Create individual plots for each model.

    Args:
        results_dir: Directory containing model results
        files_grouped: Dictionary mapping model keys to lists of files
        output_suffix: Suffix to add to output filenames
        show_std: Whether to show standard deviation in plots
        ss_len: List of suspect set lengths to include
        run_type: List of run types to include
        models: List of model names to include
    """
    for model_key, files in sorted(files_grouped.items()):
        fig, ax = plt.subplots(figsize=(6, 6))

        metric_data = {
            "real": {"p": [], "m": [], "s": []},
            "ae_synth": {"p": [], "m": [], "s": []},
            "synth": {"p": [], "m": [], "s": []},
            "correction": {"p": [], "m": [], "s": []},
        }

        for file in sorted(files, key=lambda x: parse_filename(x)["target"]):
            file_data = parse_filename(file)
            _run_type_ = file_data["run_type"]
            p = file_data["prob"]
            file_path = os.path.join(results_dir, file)
            df = pd.read_csv(file_path, delimiter="\t")

            try:
                if _run_type_ != "correction":  # and _run_type_ != "synth":
                    if "p_hat_test" in df.columns:
                        mean = df["p_hat_test"].mean()
                        std = df["p_hat_test"].std()
                elif _run_type_ == "correction":
                    if "p_hat_2MIA-comb" in df.columns:
                        mean = df["p_hat_2MIA-comb"].mean()
                        std = df["p_hat_2MIA-comb"].std()

                metric_data[_run_type_]["p"].append(p)
                metric_data[_run_type_]["m"].append(mean)
                metric_data[_run_type_]["s"].append(std)
            except Exception as e:
                continue

        # plot
        for run_type_, d in metric_data.items():
            # if run_type_ != "synth":
            if len(d["p"]) == 0:
                continue

            p = np.array(d["p"])
            m = np.array(d["m"])
            s = np.array(d["s"])
            order = np.argsort(p)

            ax.plot(
                p[order],
                m[order],
                marker="o",
                label=labels[run_type_],
                color=colors.get(run_type_, "gray"),
            )

            if show_std:
                ax.fill_between(
                    p[order],
                    m[order] - s[order],
                    m[order] + s[order],
                    color=colors.get(run_type_, "gray"),
                    alpha=0.2,
                )

        # diagonal
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("True p")
        ax.set_ylabel("Estimated p")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

        # save
        # Create a filename that includes the filtering parameters
        filter_parts = []
        filter_parts.append(f"model={target}")
        if run_type:
            filter_parts.append(f"{'-'.join(run_type)}")
        if ss_len:
            filter_parts.append(f"{'-'.join(map(str, ss_len))}")

        filter_suffix = "_".join(filter_parts)
        if filter_suffix:
            filter_suffix = f"_{filter_suffix}"

        suffix = f"_{output_suffix}" if output_suffix else ""
        out_dir = os.path.join(results_dir, "single", target)
        os.makedirs(out_dir, exist_ok=True)

        out_png = os.path.join(
            out_dir,
            f"single_{model_key}{filter_suffix[:10]}{suffix}.png",
        )
        out_pdf = os.path.join(
            out_dir,
            f"single_{model_key}{filter_suffix[:10]}{suffix}.pdf",
        )

        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)


def create_grid_plot(
    results_dir,
    target,
    files_grouped,
    output_suffix="",
    show_std=True,
    ss_len=None,
    run_type=None,
    models=None,
):
    """
    Create a grid of square plots for all models in a single figure.

    Args:
        results_dir: Directory containing model results
        files_grouped: Dictionary mapping model keys to lists of files
        output_suffix: Suffix to add to output filenames
        show_std: Whether to show standard deviation in plots
        ss_len: List of suspect set lengths to include
        run_type: List of run types to include
        models: List of model names to include
    """
    # Count the number of models
    num_models = len(files_grouped)
    if num_models == 0:
        print("No models to plot in grid.")
        return

    # Calculate grid dimensions (try to make it as square as possible)
    grid_size = int(np.ceil(np.sqrt(num_models)))
    rows = grid_size
    cols = grid_size

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)

    # Track which run types are present for the common legend
    used_run_types = set()

    # Process each model
    for i, (model_key, files) in enumerate(sorted(files_grouped.items())):
        # Calculate row and column for this subplot
        row = i // cols
        col = i % cols
        ax = axes[row, col]

        # Dictionary to store metric data for this model
        metric_data = {
            "real": {"p": [], "m": [], "s": []},
            "ae_synth": {"p": [], "m": [], "s": []},
            "synth": {"p": [], "m": [], "s": []},
            "correction": {"p": [], "m": [], "s": []},
        }

        # Process all files for this model
        for file in sorted(files, key=lambda x: parse_filename(x)["target"]):
            file_data = parse_filename(file)
            _run_type_ = file_data["run_type"]
            p = file_data["prob"]
            file_path = os.path.join(results_dir, file)
            df = pd.read_csv(file_path, delimiter="\t")

            try:
                if _run_type_ != "correction":  # and _run_type_ != "synth":
                    if "p_hat_test" in df.columns:
                        mean = df["p_hat_test"].mean()
                        std = df["p_hat_test"].std()

                elif _run_type_ == "correction":
                    if "p_hat_2MIA-comb" in df.columns:
                        mean = df["p_hat_2MIA-comb"].mean()
                        std = df["p_hat_2MIA-comb"].std()

                metric_data[_run_type_]["p"].append(p)
                metric_data[_run_type_]["m"].append(mean)
                metric_data[_run_type_]["s"].append(std)
            except Exception as e:
                continue

        # Plot each run type
        for run_type_, d in metric_data.items():
            # if run_type_ != "synth":
            if len(d["p"]) == 0:
                continue

            used_run_types.add(run_type_)
            p = np.array(d["p"])
            m = np.array(d["m"])
            s = np.array(d["s"])
            order = np.argsort(p)

            ax.plot(
                p[order],
                m[order],
                marker="o",
                color=colors.get(run_type_, "gray"),
            )

            if show_std:
                ax.fill_between(
                    p[order],
                    m[order] - s[order],
                    m[order] + s[order],
                    color=colors.get(run_type_, "gray"),
                    alpha=0.2,
                )

        # Plot diagonal line
        ax.plot([0, 1], [0, 1], "k--")
        # if "_" in model_key:
        #     parts = model_key.split("_")
        #     if len(parts) > 4 and parts[-1].isdigit() and parts[-2].isalpha():
        #         display_model_name = parts[0] + parts[1]
        #         ss_len_value = parts[2]
        #         model_type = parts[3]
        #         bins_value = parts[4]
        #         title = f"{display_model_name}\nss_len={ss_len_value}, type={model_type}, bins={bins_value}"
        #     elif len(parts) >= 2 and parts[-1].isdigit():
        #         display_model_name = "_".join(parts[:-1])
        #         ss_len_value = parts[-1]
        #         title = f"{display_model_name}\nss_len={ss_len_value}"
        #     else:
        #         title = model_key
        # else:
        #     title = model_key
        title = model_key[:25]
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("True p")
        ax.set_ylabel("Estimated p")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    # Hide empty subplots
    for i in range(num_models, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis("off")

    # Create a common legend
    handles = []
    labels_list = []
    for run_type_ in ["real", "synth", "ae_synth", "correction"]:
        if run_type_ in used_run_types:
            handle = plt.Line2D(
                [],
                [],
                color=colors[run_type_],
                marker="o",
                linestyle="-",
                label=labels[run_type_],
            )
            handles.append(handle)
            labels_list.append(labels[run_type_])

    # Add diagonal line to legend
    handles.append(plt.Line2D([], [], color="k", linestyle="--", label="True p"))
    labels_list.append("True p")
    fig.legend(
        handles,
        labels_list,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, 0.03),
        fontsize=12,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the legend
    filter_parts = []
    filter_parts.append(f"model={target}")
    if run_type:
        filter_parts.append(f"{'-'.join(run_type)}")
    if ss_len:
        filter_parts.append(f"{'-'.join(map(str, ss_len))}")

    filter_suffix = "_".join(filter_parts)
    if filter_suffix:
        filter_suffix = f"_{filter_suffix}"

    suffix = f"_{output_suffix}" if output_suffix else ""
    model_name = os.path.basename(results_dir)
    output_png = os.path.join(
        results_dir,
        "grid",
        target,
        f"grid_{model_name}{filter_suffix}{suffix}.png",
    )
    output_pdf = os.path.join(
        results_dir,
        "grid",
        target,
        f"grid_{model_name}{filter_suffix}{suffix}.pdf",
    )
    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_pdf, bbox_inches="tight")

    print(f"Grid plot created and saved to '{output_png}' and '{output_pdf}'")
    plt.close(fig)


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


def create_plots_for_model(
    results_dir,
    output_suffix="",
    ss_len=None,
    run_type=None,
    models=None,
    show_std=True,
):
    """
    Create plots for p_hat_test (real) and p_hat_test_through_scores_and_debiased for other methods

    Args:
        results_dir: Directory containing results for a specific model
        output_suffix: Suffix to add to output filenames
        ss_len: List of suspect set lengths to include
        run_type: List of run types to include
        models: List of model names to include
        show_std: Whether to show standard deviation in plots (default: True)
    """
    print(f"Processing directory: {results_dir}")

    # Check if directory exists
    if not os.path.exists(results_dir):
        print(f"Directory {results_dir} does not exist.")
        return

    # Collect all results files (excluding files with 'full' in the name)
    all_files = [
        f for f in os.listdir(results_dir) if f.endswith(".csv") and "full" not in f
    ]

    if not all_files:
        print(f"No results files found in {results_dir}.")
        return

    # Filter files based on criteria
    results_files = filter_files(all_files, ss_len, run_type, models)

    if not results_files:
        print(f"No files match the specified criteria in {results_dir}.")
        return

    # Group files by parameters
    model_files = defaultdict(dict)
    for file in results_files:
        file_data = parse_filename(file)
        key = f"{file_data['target']}_{file_data['ss_len']}_{file_data['method']}_{file_data['method_args']}"
        if key not in model_files[file_data["target"]]:
            model_files[file_data["target"]][key] = []
        model_files[file_data["target"]][key].append(file)

    # Print model files for debugging
    num_models = len(model_files)
    if num_models == 0:
        print("No valid model files found.")
        return

    # Print MAE values to console output
    print("\n" + "=" * 60)
    print("Processing models:")
    print("=" * 60)
    for model_name, files in sorted(model_files.items()):
        print(f"Model: {model_name} with {len(files)} files")
    print("=" * 60 + "\n")

    for target in list(model_files.keys()):
        create_individual_model_plots(
            results_dir,
            target,
            model_files[target],
            output_suffix=output_suffix,
            show_std=show_std,
            ss_len=ss_len,
            run_type=run_type,
            models=models,
        )
        create_grid_plot(
            results_dir,
            target,
            model_files[target],
            output_suffix=output_suffix,
            show_std=show_std,
            ss_len=ss_len,
            run_type=run_type,
            models=models,
        )


def main():
    results_dir = args.results_dir
    output_suffix = args.output_suffix
    show_std = args.show_std
    ss_len = args.ss_len
    run_type = args.run_type
    models = args.targets

    create_plots_for_model(
        results_dir,
        output_suffix=output_suffix,
        ss_len=ss_len,
        run_type=run_type,
        models=models,
        show_std=show_std,
    )


if __name__ == "__main__":
    main()
