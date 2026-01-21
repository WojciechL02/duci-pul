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


# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Create p_hat_test comparison plots for a model directory."
)
parser.add_argument(
    "model_dir",
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
    default=[500],
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
    "--bins",
    type=int,
    nargs="+",
    help="Filter results by number of bins (e.g., 5 10 30)",
)
parser.add_argument(
    "--PUL_method_type",
    type=str,
    nargs="+",
    default=["LR"],
    help="Filter results by PUL method type (e.g., LR MLP)",
)
parser.add_argument(
    "--models",
    type=str,
    nargs="+",
    default=[
        "mar_b",
        "mar_l",
        "mar_h",
        "rar_xl",
        "rar_xxl",
        "var_24",
        "var_30",
        "pythia-12b",
    ],
    help="Filter results by model names (e.g., rar_xl rar_xxl var_24 var_30)",
)
args = parser.parse_args()


# Function to extract mean value from "mean±std" format
def extract_mean(value_str):
    result = float(value_str.split("±")[0])
    return result


# Function to extract std value from "mean±std" format
def extract_std(value_str):
    parts = value_str.split("±")
    if len(parts) > 1:
        result = float(parts[1])
        return result
    return 0.0


# Function to extract model name, run type, p value, ss_len, model_type, and bins from filename
def parse_filename(filename):
    # Match pattern with model name, run_type, ss_len, bins number, PUL method name, p value
    # First try to match with specific model names to avoid matching "ae" as part of the model name
    match = re.match(
        r"results_lbe_prior_(rar_xl|rar_xxl|var_24|var_30|dit_rf|uvit_t2i_deep)_(real|ae_synth|synth|correction)_len(\d+)_bins(\d+)_lbe(\w+)_p=(.+)\.csv",
        filename,
    )
    if match:
        model_name = match.group(1)
        run_type = match.group(2)
        ss_len = int(match.group(3))
        bins = int(match.group(4))
        model_type = match.group(5)
        p_value = float(match.group(6))
        return model_name, p_value, run_type, ss_len, model_type, bins

    # If the first pattern doesn't match, try a more generic pattern
    # But be careful to avoid matching "ae_synth" as part of the model name
    match = re.match(
        r"results_lbe_prior_(.+?)_(real|ae_synth|synth|correction)_len(\d+)_bins(\d+)_lbe(\w+)_p=(.+)\.csv",
        filename,
    )
    if match:
        model_name = match.group(1)
        run_type = match.group(2)
        ss_len = int(match.group(3))
        bins = int(match.group(4))
        model_type = match.group(5)
        p_value = float(match.group(6))
        return model_name, p_value, run_type, ss_len, model_type, bins

    return None, None, None, None, None, None


def create_individual_model_plots(
    model_dir,
    files_grouped,
    output_suffix="",
    show_std=True,
    ss_len=None,
    run_type=None,
    bins=None,
    PUL_method_type=None,
    models=None,
):
    """
    Create individual plots for each model.

    Args:
        model_dir: Directory containing model results
        files_grouped: Dictionary mapping model keys to lists of files
        output_suffix: Suffix to add to output filenames
        show_std: Whether to show standard deviation in plots
        ss_len: List of suspect set lengths to include
        run_type: List of run types to include
        bins: List of bin numbers to include
        PUL_method_type: List of PUL method types to include
        models: List of model names to include
    """
    for model_key, files in sorted(files_grouped.items()):
        fig, ax = plt.subplots(figsize=(6, 6))

        # same colors as main plot
        colors = {
            "real": "blue",
            "ae_synth": "red",
            "synth": "green",
            "correction": "gray",
        }

        metric_data = {
            "real": {"p": [], "m": [], "s": []},
            "ae_synth": {"p": [], "m": [], "s": []},
            "synth": {"p": [], "m": [], "s": []},
            "correction": {"p": [], "m": [], "s": []},
        }

        # process all files for this model
        for file in sorted(files, key=lambda x: parse_filename(x)[1]):
            _, p, _run_type_, _, _, _ = parse_filename(file)
            file_path = os.path.join(model_dir, file)
            df = pd.read_csv(file_path, delimiter="\t")

            try:
                if _run_type_ != "correction" and _run_type_ != "synth":
                    if "p_hat_test_mean_std" in df.columns:
                        val = df["p_hat_test_mean_std"].iloc[0]
                        mean = extract_mean(val)
                        std = extract_std(val)
                elif _run_type_ == "correction":
                    if "p_hat_2MIA-comb_mean_std" in df.columns:
                        val = df["p_hat_2MIA-comb_mean_std"].iloc[0]
                        mean = extract_mean(val)
                        std = extract_std(val)

                metric_data[_run_type_]["p"].append(p)
                metric_data[_run_type_]["m"].append(mean)
                metric_data[_run_type_]["s"].append(std)
            except Exception as e:
                continue

        # plot
        labels = {
            "real": "Real",
            "ae_synth": "Synthetic (no correction)",
            "correction": "Synthetic (+correction) (ours)",
        }
        for run_type_, d in metric_data.items():
            if run_type_ != "synth":
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

        # ax.set_title(f"Model: {model_key}", fontsize=16, fontweight="bold")
        ax.set_xlabel("True p")
        ax.set_ylabel("Estimated p")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

        # save
        # Create a filename that includes the filtering parameters
        filter_parts = []
        if ss_len:
            filter_parts.append(f"ss_len={'-'.join(map(str, ss_len))}")
        if run_type:
            filter_parts.append(f"run_type={'-'.join(run_type)}")
        if bins:
            filter_parts.append(f"bins={'-'.join(map(str, bins))}")
        if PUL_method_type:
            filter_parts.append(f"PUL_method={'-'.join(PUL_method_type)}")
        if models:
            filter_parts.append(f"models={'-'.join(models)}")

        filter_suffix = "_".join(filter_parts)
        if filter_suffix:
            filter_suffix = f"_{filter_suffix}"

        suffix = f"_{output_suffix}" if output_suffix else ""
        out_png = os.path.join(
            model_dir, f"p_hat_test_single_{model_key}{filter_suffix}{suffix}.png"
        )
        out_pdf = os.path.join(
            model_dir, f"p_hat_test_single_{model_key}{filter_suffix}{suffix}.pdf"
        )

        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)


def create_grid_plot(
    model_dir,
    files_grouped,
    output_suffix="",
    show_std=True,
    ss_len=None,
    run_type=None,
    bins=None,
    PUL_method_type=None,
    models=None,
):
    """
    Create a grid of square plots for all models in a single figure.

    Args:
        model_dir: Directory containing model results
        files_grouped: Dictionary mapping model keys to lists of files
        output_suffix: Suffix to add to output filenames
        show_std: Whether to show standard deviation in plots
        ss_len: List of suspect set lengths to include
        run_type: List of run types to include
        bins: List of bin numbers to include
        PUL_method_type: List of PUL method types to include
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

    # Create a figure with subplots in a grid
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)

    # Define colors for different run types (same as individual plots)
    colors = {
        "real": "blue",
        "ae_synth": "red",
        "synth": "green",
        "correction": "gray",
    }

    # Define labels (same as individual plots)
    labels = {
        "real": "Real",
        "ae_synth": "Synthetic (no correction)",
        "correction": "Synthetic (+correction) (ours)",
    }

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
        for file in sorted(files, key=lambda x: parse_filename(x)[1]):
            _, p, _run_type_, _, _, _ = parse_filename(file)
            file_path = os.path.join(model_dir, file)
            df = pd.read_csv(file_path, delimiter="\t")

            try:
                if _run_type_ != "correction" and _run_type_ != "synth":
                    if "p_hat_test_mean_std" in df.columns:
                        val = df["p_hat_test_mean_std"].iloc[0]
                        mean = extract_mean(val)
                        std = extract_std(val)

                        # Debug print for ae_synth
                elif _run_type_ == "correction":
                    if "p_hat_2MIA-comb_mean_std" in df.columns:
                        val = df["p_hat_2MIA-comb_mean_std"].iloc[0]
                        mean = extract_mean(val)
                        std = extract_std(val)

                metric_data[_run_type_]["p"].append(p)
                metric_data[_run_type_]["m"].append(mean)
                metric_data[_run_type_]["s"].append(std)
            except Exception as e:
                continue

        # Plot each run type
        for run_type_, d in metric_data.items():
            if run_type_ != "synth":
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
                    # Don't include label here, we'll create a common legend
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

        # Set title and labels
        if "_" in model_key:
            parts = model_key.split("_")
            if len(parts) > 4 and parts[-1].isdigit() and parts[-2].isalpha():
                display_model_name = parts[0] + parts[1]
                ss_len_value = parts[2]
                model_type = parts[3]
                bins_value = parts[4]
                title = f"{display_model_name}\nss_len={ss_len_value}, type={model_type}, bins={bins_value}"
            elif len(parts) >= 2 and parts[-1].isdigit():
                display_model_name = "_".join(parts[:-1])
                ss_len_value = parts[-1]
                title = f"{display_model_name}\nss_len={ss_len_value}"
            else:
                title = model_key
        else:
            title = model_key

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
    for run_type_ in ["real", "ae_synth", "correction"]:
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

    # Place the legend at the bottom of the figure
    fig.legend(
        handles,
        labels_list,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, 0.03),
        fontsize=12,
    )

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the legend

    # Save the figure
    # Create a filename that includes the filtering parameters
    filter_parts = []
    if ss_len:
        filter_parts.append(f"ss_len={'-'.join(map(str, ss_len))}")
    if run_type:
        filter_parts.append(f"run_type={'-'.join(run_type)}")
    if bins:
        filter_parts.append(f"bins={'-'.join(map(str, bins))}")
    if PUL_method_type:
        filter_parts.append(f"PUL_method={'-'.join(PUL_method_type)}")
    if models:
        filter_parts.append(f"models={'-'.join(models)}")

    filter_suffix = "_".join(filter_parts)
    if filter_suffix:
        filter_suffix = f"_{filter_suffix}"

    suffix = f"_{output_suffix}" if output_suffix else ""
    model_name = os.path.basename(model_dir)
    output_png = os.path.join(
        model_dir, f"p_hat_test_grid_{model_name}{filter_suffix}{suffix}.png"
    )
    output_pdf = os.path.join(
        model_dir, f"p_hat_test_grid_{model_name}{filter_suffix}{suffix}.pdf"
    )
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_pdf, bbox_inches="tight")

    print(f"Grid plot created and saved to '{output_png}' and '{output_pdf}'")
    plt.close(fig)


def filter_files(
    files, ss_len=None, run_type=None, bins=None, PUL_method_type=None, models=None
):
    """
    Filter files based on specified criteria

    Args:
        files: List of files to filter
        ss_len: List of suspect set lengths to include
        run_type: List of run types to include
        bins: List of bin numbers to include
        PUL_method_type: List of PUL method types to include
        models: List of model names to include

    Returns:
        List of files that match the criteria
    """
    filtered_files = []

    for file in files:
        model_name, p_value, file_run_type, file_ss_len, file_model_type, file_bins = (
            parse_filename(file)
        )

        # Skip files that don't match the pattern
        if model_name is None:
            continue

        # Filter by model name if specified
        if models is not None and model_name not in models:
            continue

        # Filter by ss_len if specified
        if ss_len is not None and file_ss_len not in ss_len:
            continue

        # Filter by run_type if specified
        if run_type is not None and file_run_type not in run_type:
            continue

        # Filter by bins if specified
        if bins is not None and file_bins not in bins:
            continue

        # Filter by PUL_method_type if specified
        if PUL_method_type is not None and file_model_type not in PUL_method_type:
            continue

        filtered_files.append(file)

    return filtered_files


def create_plots_for_model(
    model_dir,
    output_suffix="",
    ss_len=None,
    run_type=None,
    bins=None,
    PUL_method_type=None,
    models=None,
    show_std=True,
):
    """
    Create plots for p_hat_test (real) and p_hat_test_through_scores_and_debiased for other methods

    Args:
        model_dir: Directory containing results for a specific model
        output_suffix: Suffix to add to output filenames
        ss_len: List of suspect set lengths to include
        run_type: List of run types to include
        bins: List of bin numbers to include
        PUL_method_type: List of PUL method types to include
        models: List of model names to include
        show_std: Whether to show standard deviation in plots (default: True)
    """
    print(f"Processing directory: {model_dir}")

    # Check if directory exists
    if not os.path.exists(model_dir):
        print(f"Directory {model_dir} does not exist.")
        return

    # Collect all results files (excluding files with 'full' in the name)
    all_files = [
        f
        for f in os.listdir(model_dir)
        if f.startswith("results_lbe_prior_") and f.endswith(".csv") and "full" not in f
    ]

    if not all_files:
        print(f"No results files found in {model_dir}.")
        return

    # Filter files based on criteria
    results_files = filter_files(
        all_files, ss_len, run_type, bins, PUL_method_type, models
    )

    if not results_files:
        print(f"No files match the specified criteria in {model_dir}.")
        return

    # Group files by model, ss_len, model_type, and bins
    model_files = defaultdict(list)
    for file in results_files:
        model_name, _, file_run_type, file_ss_len, file_model_type, file_bins = (
            parse_filename(file)
        )
        if model_name:
            # Create a key based on available information
            if (
                file_ss_len is not None
                and file_model_type is not None
                and file_bins is not None
            ):
                key = f"{model_name}_{file_ss_len}_{file_model_type}_{file_bins}"
            elif file_ss_len is not None:
                key = f"{model_name}_{file_ss_len}"
            else:
                key = model_name
            model_files[key].append(file)

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

    # Create individual plots for each model
    create_individual_model_plots(
        model_dir,
        model_files,
        output_suffix=output_suffix,
        show_std=show_std,
        ss_len=ss_len,
        run_type=run_type,
        bins=bins,
        PUL_method_type=PUL_method_type,
        models=models,
    )

    # Create a grid plot with all models
    create_grid_plot(
        model_dir,
        model_files,
        output_suffix=output_suffix,
        show_std=show_std,
        ss_len=ss_len,
        run_type=run_type,
        bins=bins,
        PUL_method_type=PUL_method_type,
        models=models,
    )


def main():
    # Get command line arguments
    model_dir = args.model_dir
    output_suffix = args.output_suffix
    show_std = args.show_std
    ss_len = args.ss_len
    run_type = args.run_type
    bins = args.bins
    PUL_method_type = args.PUL_method_type
    models = args.models

    # Process the specified directory
    create_plots_for_model(
        model_dir,
        output_suffix=output_suffix,
        ss_len=ss_len,
        run_type=run_type,
        bins=bins,
        PUL_method_type=PUL_method_type,
        models=models,
        show_std=show_std,
    )


if __name__ == "__main__":
    main()
