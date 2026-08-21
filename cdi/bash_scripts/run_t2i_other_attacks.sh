#!/usr/bin/env bash
# Generate carlini_lt, multiple_loss, and gradient_masking features for the three
# T2I targets (minFM, DeltaFM, NitroT) on real data, with the same 2k-eval + 2k-clf
# layout used for CLiD. Runs sequentially on a single GPU.
#
# Usage:
#   GPU=0 bash cdi/bash_scripts/run_t2i_other_attacks.sh
#
# Env overrides:
#   GPU       (default 0)
#   N_EVAL    (default 1000)
#   N_CLF     (default 1000)
#   ATTACKS   (default "carlini_lt multiple_loss gradient_masking")
#   MODELS    (default "minfm deltafm nitrot")
set -euo pipefail

GPU="${GPU:-0}"
N_EVAL="${N_EVAL:-1000}"
N_CLF="${N_CLF:-1000}"
ATTACKS="${ATTACKS:-carlini_lt multiple_loss gradient_masking}"
MODELS="${MODELS:-minfm deltafm nitrot}"
CDI_DIR="/home/jdubinsk/dataset_inference_dm/duci-pul/cdi"
VENV="/home/jdubinsk/dataset_inference_dm/duci-pul/.venv/bin/python3"

cd "$CDI_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"

run_one() {
    local model_key="$1"
    local attack="$2"
    local split="$3"

    case "$model_key" in
        minfm)
            local model="minfm_flux_tiny" ; local dataset="imagenet" ; local run_id="t2i1k"
            local bs=8 ; local extra="++dataset.dataset_path=/data/imagenet_1k" ;;
        deltafm)
            local model="deltafm_sitxl"   ; local dataset="imagenet" ; local run_id="t2i1k"
            local bs=8 ; local extra="++dataset.dataset_path=/data/imagenet_1k" ;;
        nitrot)
            local model="nitrot_1p2b"     ; local dataset="cc12m_nitrot" ; local run_id="cc12m1k"
            local bs=4 ; local extra="" ;;
        *) echo "unknown model_key: $model_key" >&2 ; exit 1 ;;
    esac

    local out_file="$CDI_DIR/out_t2i_mia/features/${model}_${attack}_${run_id}_${dataset}_${split}.npz"
    if [ -f "$out_file" ]; then
        echo "[$(date)] SKIP existing $out_file"
        return 0
    fi

    echo "[$(date)] $model | $dataset/$split | $attack | bs=$bs"
    "$VENV" main.py \
        +model="$model" \
        +dataset="$dataset" \
        +attack="$attack" \
        +action=features_extraction \
        $extra \
        ++dataset.split="$split" \
        ++cfg.n_samples_eval="$N_EVAL" \
        ++cfg.train_samples="$N_CLF" \
        ++cfg.run_id="$run_id" \
        ++cfg.path_to_features=./out_t2i_mia/features \
        ++cfg.path_to_scores=./out_t2i_mia/scores \
        ++cfg.dataloader_num_workers=4 \
        ++model.batch_size="$bs" \
        ++action.device="cuda:0"
}

splits_for() {
    case "$1" in
        nitrot) echo "cc12mhi cc12mlo" ;;
        *)      echo "train val" ;;
    esac
}

for model_key in $MODELS; do
    for attack in $ATTACKS; do
        for split in $(splits_for "$model_key"); do
            run_one "$model_key" "$attack" "$split"
        done
    done
done

echo "[$(date)] all done"
