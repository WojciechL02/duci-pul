#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
from collections import defaultdict

# Parse command line arguments
parser = argparse.ArgumentParser(description='Create p_hat_test comparison plots for different models and methods.')
parser.add_argument('--results_dir', type=str, default="../",
                    help='Base directory containing results (default: ../)')
parser.add_argument('--bins', type=str, default="10",
                    help='Bin size to use (5, 10, or 30) (default: 10)')
parser.add_argument('--output_suffix', type=str, default="",
                    help='Suffix to add to output filenames')
args = parser.parse_args()

# Function to extract mean value from "mean±std" format
def extract_mean(value_str):
    return float(value_str.split('±')[0])

# Function to extract model name, run type, and p value from filename
def parse_filename(filename):
    # Try to match lbe_prior pattern with run_type
    match = re.match(r'results_lbe_prior_(rar_xl|rar_xxl|var_24|var_30)_(real|ae_synth|synth|correction)_p=(.+)\.csv', filename)
    if match:
        model_name = match.group(1)
        run_type = match.group(2)
        p_value = float(match.group(3))
        return model_name, p_value, run_type

    # Try to match lbe_prior pattern without run_type (assume it's "real")
    match = re.match(r'results_lbe_prior_(rar_xl|rar_xxl|var_24|var_30)_p=(.+)\.csv', filename)
    if match:
        model_name = match.group(1)
        p_value = float(match.group(2))
        return model_name, p_value, "real"

    # Try to match agg pattern (assume it's "real")
    match = re.match(r'results_agg_(rar_xl|rar_xxl|var_24|var_30)_p=(.+)\.csv', filename)
    if match:
        model_name = match.group(1)
        p_value = float(match.group(2))
        return model_name, p_value, "real"

    return None, None, None

def create_plots_for_model(model_dir, output_suffix="", bins=None):
    """
    Create plots for p_hat_test (real) and p_hat_test_through_scores_and_debiased for other methods

    Args:
        model_dir: Directory containing results for a specific model
        output_suffix: Suffix to add to output filenames
        bins: Number of bins used in the model
    """
    print(f"Processing directory: {model_dir}")

    # Check if directory exists
    if not os.path.exists(model_dir):
        print(f"Directory {model_dir} does not exist.")
        return

    # Collect all results files (excluding files with 'full' in the name)
    results_files = [f for f in os.listdir(model_dir) if (f.startswith('results_lbe_prior_') or f.startswith('results_agg_')) and f.endswith('.csv') and 'full' not in f]

    if not results_files:
        print(f"No results files found in {model_dir}.")
        return

    # Group files by model
    model_files = defaultdict(list)
    for file in results_files:
        model_name, _, _ = parse_filename(file)
        if model_name:
            model_files[model_name].append(file)

    # Create a figure with subplots for each model
    num_models = len(model_files)
    if num_models == 0:
        print("No valid model files found.")
        return

    fig, axes = plt.subplots(num_models, 1, figsize=(14, 5 * num_models), sharex=True)

    # If there's only one model, make axes iterable
    if num_models == 1:
        axes = [axes]

    # Define colors for different run types
    colors = {
        'real': 'blue',
        'ae_synth': 'red',
        'synth': 'green',
        'corrected': 'purple'
    }

    # Process each model
    for i, (model_name, files) in enumerate(sorted(model_files.items())):
        ax = axes[i]

        # Dictionary to store metric data across p values for each run type
        metric_data = {
            'real': {'p_values': [], 'metric_values': []},
            'ae_synth': {'p_values': [], 'metric_values': []},
            'synth': {'p_values': [], 'metric_values': []},
            'correction': {'p_values': [], 'metric_values': []}
        }

        print(f"\nProcessing model: {model_name} with {len(files)} files")

        # Process each file for this model
        for file in sorted(files, key=lambda x: parse_filename(x)[1]):
            _, p_value, run_type = parse_filename(file)
            print(f"  Processing file: {file} with p_value={p_value}, run_type={run_type}")

            try:
                # Read the CSV file
                file_path = os.path.join(model_dir, file)
                print(f"  Reading file from: {file_path}")
                df = pd.read_csv(file_path, delimiter='\t')

                if run_type == "real":
                    # For real, use p_hat_test
                    if "p_hat_test_mean_std" in df.columns:
                        p_hat_test_value = extract_mean(df["p_hat_test_mean_std"].iloc[0])
                        metric_data['real']['p_values'].append(p_value)
                        metric_data['real']['metric_values'].append(p_hat_test_value)
                        print(f"    Extracted p_hat_test value: {p_hat_test_value} for p={p_value} (real)")
                else:
                    # For ae_synth, synth, corrected, use p_hat_test_through_scores_and_debised
                    if "p_hat_through_scores_and_debiased_mean_std" in df.columns:
                        p_hat_through_scores_value = extract_mean(df["p_hat_through_scores_and_debiased_mean_std"].iloc[0])
                        metric_data[run_type]['p_values'].append(p_value)
                        metric_data[run_type]['metric_values'].append(p_hat_through_scores_value)
                        print(f"    Extracted p_hat_through_scores_and_debiased value: {p_hat_through_scores_value} for p={p_value} ({run_type})")
            except Exception as e:
                print(f"  Error processing {file}: {e}")

        # Plot each run type
        for run_type, data in metric_data.items():
            if len(data['p_values']) > 0:
                # Sort by p_values to ensure correct line plotting
                p_values = np.array(data['p_values'])
                metric_values = np.array(data['metric_values'])
                sort_idx = np.argsort(p_values)

                ax.plot(p_values[sort_idx], metric_values[sort_idx], marker='o', 
                       label=run_type, color=colors.get(run_type, 'gray'), linewidth=2)

        # Plot the true p values (diagonal line)
        ax.plot([0.0, 1.0], [0.0, 1.0], 'k--', label='True p', linewidth=2)

        # Set title and labels
        ax.set_title(f'Model: {model_name}', fontsize=16, fontweight='bold')
        ax.set_ylabel('Estimated p', fontsize=14)
        ax.set_xlabel('True p value', fontsize=14)
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)

        # Set axis limits
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

        # Add minor grid
        ax.minorticks_on()
        ax.grid(which='minor', linestyle=':', alpha=0.4)

    # Set common x-label for the entire figure
    fig.text(0.5, 0.01, 'True p value', ha='center', fontsize=16, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.subplots_adjust(hspace=0.3)

    # Add a title for the entire figure
    model_type = "Linear Regression" if "LR" in model_dir else "MLP"
    bin_info = f" ({bins} bins)" if bins else ""
    title = f'Comparison of real, synth, ae_synth, corrected for {model_type}{bin_info} Models'
    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.99)

    # Save the figure
    suffix = f"_{output_suffix}" if output_suffix else ""
    model_name = os.path.basename(model_dir)
    output_png = os.path.join(model_dir, f'p_hat_test_comparison_plots_{model_name}{suffix}.png')
    output_pdf = os.path.join(model_dir, f'p_hat_test_comparison_plots_{model_name}{suffix}.pdf')
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, bbox_inches='tight')

    print(f"Plots created and saved to '{output_png}' and '{output_pdf}'")

def main():
    # Get command line arguments
    base_dir = args.results_dir
    bins = args.bins
    output_suffix = args.output_suffix

    # Define directories to process
    directories = []

    # Add r11_11bin10_LR and r11_11bin10 directories for bins10
    if bins == "10":
        directories.extend([
            os.path.join(base_dir, "r11_11_bin10_LR"),
            os.path.join(base_dir, "r11_11_bin10")
        ])
    # Add similar directories for bins5
    elif bins == "5":
        directories.extend([
            os.path.join(base_dir, "r11_11_bin5_LR"),
            os.path.join(base_dir, "r11_11_bin5")
        ])
    # Add similar directories for bins30
    elif bins == "30":
        directories.extend([
            os.path.join(base_dir, "r11_11_bin30_LR"),
            os.path.join(base_dir, "r11_11_bin30")
        ])
    else:
        print(f"Invalid bins value: {bins}. Please use 5, 10, or 30.")
        return

    # Process each directory
    for directory in directories:
        create_plots_for_model(directory, output_suffix, bins)

if __name__ == "__main__":
    main()
