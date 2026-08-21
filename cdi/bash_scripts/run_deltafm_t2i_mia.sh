#!/usr/bin/env bash
set -euo pipefail

SPLIT="${SPLIT:-train}"
GPU="${GPU:-0}"
N_EVAL="${N_EVAL:-1000}"
N_CLF="${N_CLF:-1000}"
BS="${BS:-8}"
RUN_ID="${RUN_ID:-t2i1k}"
CDI_DIR="/home/jdubinsk/dataset_inference_dm/duci-pul/cdi"
VENV="/home/jdubinsk/dataset_inference_dm/duci-pul/.venv/bin/python3"

cd "$CDI_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"
echo "[$(date)] DeltaFM CLiD features | split=$SPLIT gpu=$GPU n_eval=$N_EVAL n_clf=$N_CLF bs=$BS"

"$VENV" main.py \
    +model=deltafm_sitxl \
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

echo "[$(date)] features done split=$SPLIT"
