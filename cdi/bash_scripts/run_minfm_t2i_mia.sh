#!/usr/bin/env bash
# Usage:
#   SPLIT=train GPU=0 bash cdi/bash_scripts/run_minfm_t2i_mia.sh
#   SPLIT=val   GPU=1 bash cdi/bash_scripts/run_minfm_t2i_mia.sh
set -euo pipefail

SPLIT="${SPLIT:-train}"
GPU="${GPU:-0}"
# We need `train_samples` labeled examples for the LR classifier + `n_samples_eval`
# held-out examples for the per-sample CLiD score used in DI bootstrap.
N_EVAL="${N_EVAL:-1000}"
N_CLF="${N_CLF:-1000}"
BS="${BS:-8}"
RUN_ID="${RUN_ID:-t2i_1keval_1kclf}"
CDI_DIR="/home/jdubinsk/dataset_inference_dm/duci-pul/cdi"
VENV="/home/jdubinsk/dataset_inference_dm/duci-pul/.venv/bin/python3"

cd "$CDI_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"

TOTAL=$((N_EVAL + N_CLF))
echo "[$(date)] minFM CLiD features | split=$SPLIT gpu=$GPU n_eval=$N_EVAL n_clf=$N_CLF total=$TOTAL bs=$BS"

"$VENV" main.py \
    +model=minfm_flux_tiny \
    +dataset=imagenet \
    +attack=clid \
    +action=features_extraction \
    ++dataset.dataset_path=/data/imagenet_1k \
    ++dataset.split="$SPLIT" \
    ++cfg.n_samples_eval="$N_EVAL" \
    ++cfg.train_samples="$N_CLF" \
    ++cfg.run_id="$RUN_ID" \
    ++cfg.path_to_features=./out_t2i_mia/features \
    ++cfg.path_to_scores=./out_t2i_mia/scores \
    ++cfg.dataloader_num_workers=4 \
    ++model.batch_size="$BS" \
    ++action.device="cuda:0"

echo "[$(date)] features done for split=$SPLIT"
