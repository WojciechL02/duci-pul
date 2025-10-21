import argparse
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Parse command line arguments
parser = argparse.ArgumentParser(description='Create model comparison plots from results files.')
parser.add_argument('--results_dir', type=str, default="../results", 
                    help='Directory containing results files (default: ../results)')
args = parser.parse_args()

# Directory containing results
results_dir = args.results_dir

# Function to create plots for a specific metric
def create_plots(metric_name, output_suffix=""):
    """
    Create plots for a specific metric (p_hat or lowerbound)

    Args:
        metric_name: Name of the metric to plot (e.g., 'p_hat' or 'lowerbound')
        output_suffix: Suffix to add to output filenames
    """

# Function to extract mean value from "mean±std" format
def extract_mean(value_str):
    return float(value_str.split('±')[0])

# Function to extract model name and p value from filename
def parse_filename(filename):
    match = re.match(r'results_agg_(.+)_p=(.+)\.csv', filename)
    if match:
        model_name = match.group(1)
        p_value = float(match.group(2))
        return model_name, p_value
    return None, None

def create_plots(metric_name, output_suffix=""):
    """
    Create plots for a specific metric (p_hat or lowerbound)

    Args:
        metric_name: Name of the metric to plot (e.g., 'p_hat' or 'lowerbound')
        output_suffix: Suffix to add to output filenames
    """
    # Collect all results files
    results_files = [f for f in os.listdir(results_dir) if f.startswith('results_agg_') and f.endswith('.csv')]

    # Group files by model
    model_files = defaultdict(list)
    for file in results_files:
        model_name, _ = parse_filename(file)
        if model_name:
            model_files[model_name].append(file)

    # Get all unique methods across all files to ensure consistent colors
    all_methods = set()
    for model_name, files in model_files.items():
        for file in files:
            try:
                df = pd.read_csv(os.path.join(results_dir, file), delimiter='\t')
                all_methods.update(df['method'].unique())
            except Exception as e:
                print(f"Error reading {file}: {e}")

    # Create a color map for methods
    colors = plt.cm.tab10.colors + plt.cm.tab20.colors
    method_colors = {method: colors[i % len(colors)] for i, method in enumerate(sorted(all_methods))}

    # Create a figure with subplots for each model
    num_models = len(model_files)
    fig, axes = plt.subplots(num_models, 1, figsize=(14, 5 * num_models), sharex=True)

    # If there's only one model, make axes iterable
    if num_models == 1:
        axes = [axes]

    # Process each model
    for i, (model_name, files) in enumerate(sorted(model_files.items())):
        ax = axes[i]

        # Dictionary to store method data across p values
        method_data = defaultdict(lambda: {'p_values': [], 'metric_values': []})

        # Process each file for this model
        for file in sorted(files, key=lambda x: parse_filename(x)[1]):
            _, p_value = parse_filename(file)

            try:
                # Read the CSV file
                df = pd.read_csv(os.path.join(results_dir, file), delimiter='\t')

                # Extract metric values for each method
                for _, row in df.iterrows():
                    method = row['method']
                    if f"{metric_name}_mean_std" in row:
                        metric_value = extract_mean(row[f"{metric_name}_mean_std"])
                    else:
                        # Skip if the metric is not available for this method
                        continue

                    method_data[method]['p_values'].append(p_value)
                    method_data[method]['metric_values'].append(metric_value)
            except Exception as e:
                print(f"Error processing {file}: {e}")

        # Plot each method
        for method, data in sorted(method_data.items()):
            if len(data['p_values']) > 0:
                # Sort by p_values to ensure correct line plotting
                p_values = np.array(data['p_values'])
                metric_values = np.array(data['metric_values'])
                sort_idx = np.argsort(p_values)

                ax.plot(p_values[sort_idx], metric_values[sort_idx], marker='o', 
                       label=method, color=method_colors.get(method), linewidth=2)

        # Plot the true p values (diagonal line)
        if method_data:
            p_min = min([min(data['p_values']) for data in method_data.values() if data['p_values']])
            p_max = max([max(data['p_values']) for data in method_data.values() if data['p_values']])
            ax.plot([p_min, p_max], [p_min, p_max], 'k--', label='True p', linewidth=2)

        # Set title and labels
        ax.set_title(f'Model: {model_name}', fontsize=16, fontweight='bold')
        metric_label = 'Estimated p (p_hat)' if metric_name == 'p_hat' else f'Estimated {metric_name}'
        ax.set_ylabel(metric_label, fontsize=14)
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
    title = 'Comparison of Methods Across Different Models'
    if metric_name != 'p_hat':
        title += f' ({metric_name})'
    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.99)

    # Save the figure
    suffix = f"_{output_suffix}" if output_suffix else ""
    output_png = os.path.join(results_dir, f'model_comparison_plots{suffix}.png')
    output_pdf = os.path.join(results_dir, f'model_comparison_plots{suffix}.pdf')
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, bbox_inches='tight')

    print(f"Plots created and saved to '{output_png}' and '{output_pdf}'")

# Create plots for p_hat (original behavior)
create_plots('p_hat')

# Create plots for lowerbound
create_plots('lowerbound', 'lowerbound')
