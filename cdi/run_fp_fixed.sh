#!/bin/bash
set -e

P=/home/jdubinsk/dataset_inference_dm/duci-pul/.venv/bin/python3
GPU="cuda:2"
OUT_F="./out_fixed/features"
OUT_S="./out_fixed/scores"
COMMON="++cfg.path_to_features=$OUT_F ++cfg.path_to_scores=$OUT_S ++action.device=$GPU"

DATASETS=(fp_infinity_last2 fp_infinity_last4 fp_var_last6 fp_var_last4)
ATTACKS=(clid gradient_masking multiple_loss carlini_lt)

cd /home/jdubinsk/dataset_inference_dm/duci-pul/cdi

for DS in "${DATASETS[@]}"; do
  for SPLIT in train val; do
    for ATK in "${ATTACKS[@]}"; do
      echo "=== features: $DS $SPLIT $ATK ==="
      $P main.py +model=dit_rf +action=features_extraction +attack=$ATK +dataset=$DS \
        ++dataset.split=$SPLIT $COMMON
    done

    echo "=== scores: $DS $SPLIT carlini_lt ==="
    $P main.py +model=dit_rf +action=scores_computation +attack=carlini_lt +dataset=$DS \
      ++dataset.split=$SPLIT $COMMON
  done

  for SPLIT in train val; do
    echo "=== combination: $DS $SPLIT ==="
    $P main.py +model=dit_rf +action=features_extraction +attack=combination_attack +dataset=$DS \
      ++dataset.split=$SPLIT $COMMON
  done
done

echo "=== FP ALL DONE ==="
