#!/usr/bin/env python3
import numpy as np
import os
import subprocess

# Create results directory if it doesn't exist
results_dir = "../r11_11_bin5"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Created results directory: {results_dir}")

# Models to run
models = ["rar_xl", "rar_xxl", "var_24", "var_30"]

# Probability values from 0.0 to 1.0 in increments of 0.1
prob_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
run_types=["real", "synth", "ae_synth", "correction"]
# Number of iterations
nsym = 5

# LBE model to use
lbe_model = "MLP"

# Number of bins
bins = 5

# Device to use
device = 6

# Total number of experiments
total_experiments = len(models) * len(prob_values) * len(run_types)
completed_experiments = 0

print(f"Starting {total_experiments} experiments...")

# Loop through models and probability values
for run_type in run_types:
    for model in models:
        for prob in prob_values:
            print(f"\n{'='*80}")
            print(f"Running experiment for model: {model}, probability: {prob:.1f}")
            print(f"Progress: {completed_experiments}/{total_experiments} ({completed_experiments/total_experiments*100:.1f}%)")
            print(f"{'='*80}\n")

            # Build command
            cmd = [
                "python", "lbe_with_prior_fit_on_suspectset.py",
                "-nsym", str(nsym),
                "-prob", str(prob),
                "-lbe_model", lbe_model,
                "-data", model,
                "-run_type", run_type,
                "-results", results_dir,
                "-bins", str(bins),
                "-device", str(device)
            ]

            # Run command
            try:
                subprocess.run(cmd, check=True, cwd=".")
                print(f"Successfully completed experiment for model: {model}, probability: {prob:.1f}")
            except subprocess.CalledProcessError as e:
                print(f"Error running experiment for model: {model}, probability: {prob:.1f}")
                print(f"Error: {e}")

            completed_experiments += 1

print(f"\nAll experiments completed. Results saved to {results_dir}")