#!/usr/bin/env bash

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed"
    exit 1
fi

# Array of all available models
models=("var_16" "var_20" "var_24" "var_30" "rar_b" "rar_l" "rar_xl" "rar_xxl" "mar_b" "mar_l" "mar_h")

# Loop through probability values from 0 to 0.9 with 0.1 increments
for p in $(seq 0 0.1 0.9); do
    echo "Running with probability p=$p"
    
    # Loop through all models
    for model in "${models[@]}"; do
        echo "  Model: $model"
        if ! python3 test_pul_methods.py -data $model -nsym 5 -prob $p; then
            echo "Error running model $model with probability $p"
            # Uncomment next line if you want to stop on first error
             exit 1
        fi
    done
done

echo "All experiments completed!"