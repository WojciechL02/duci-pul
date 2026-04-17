#!/bin/bash
# Canonical end-to-end CDI feature/score runner.
# Loops over every MODEL and DATASET combination and produces the full
# pipeline needed for combination_attack:
#   Phase 1 (GPU): features for carlini_lt, gradient_masking, multiple_loss, clid
#   Phase 2 (CPU): scores_computation for carlini_lt
#   Phase 3 (CPU): features_extraction for combination_attack
#   Phase 4 (CPU): scores_computation for combination_attack
#
# Defaults cover real (ImageNet) + TP + FP at last-4 scale for the DiT-RF
# family. Any env var below can be overridden before invocation, e.g.
#   MODELS="dit_rf_g" DATASETS="imagenet_5pc" bash cdi/bash_scripts/run_synthetic_cdi.sh
#
# Env overrides:
#   MODELS             space-separated model names  (default: "dit_rf dit_rf_g")
#   DATASETS           space-separated dataset names (default: "imagenet_5pc tp_var_last4 fp_var_last4")
#   SPLITS             space-separated splits       (default: "train val")
#   GPU_ATTACKS        space-separated GPU attacks  (default: "carlini_lt gradient_masking multiple_loss clid")
#   GPUS               space-separated GPU indices  (default: "0 1 2")
#   OUT_ROOT           output root directory        (default: "<cdi>/out")
#   IMAGENET_PATH      absolute path to real ImageNet, used when 'imagenet_5pc' or 'imagenet' is in DATASETS
#   PYTHON_BIN         python interpreter path      (default: <repo>/.venv/bin/python3)

set -euo pipefail

CDI_DIR="$(cd "$(dirname "$0")/.." && pwd)"
P="${PYTHON_BIN:-$CDI_DIR/../.venv/bin/python3}"

MODELS=(${MODELS:-dit_rf dit_rf_g})
DATASETS=(${DATASETS:-imagenet_5pc tp_var_last4 fp_var_last4})
SPLITS=(${SPLITS:-train val})
GPU_ATTACKS=(${GPU_ATTACKS:-carlini_lt gradient_masking multiple_loss clid})
GPUS=(${GPUS:-0 1 2})
NUM_GPUS=${#GPUS[@]}

OUT_ROOT="${OUT_ROOT:-$CDI_DIR/out}"
OUT_F="$OUT_ROOT/features"
OUT_S="$OUT_ROOT/scores"
LOG_DIR="$CDI_DIR/logs"
mkdir -p "$OUT_F" "$OUT_S" "$LOG_DIR"

export PYTHONDONTWRITEBYTECODE=1

# Optional real-ImageNet path override (only applied when imagenet_5pc / imagenet in DATASETS)
dataset_overrides() {
    local dataset="$1"
    case "$dataset" in
        imagenet_5pc|imagenet)
            if [ -n "${IMAGENET_PATH:-}" ]; then
                echo "++dataset.dataset_path=$IMAGENET_PATH"
            fi
            ;;
    esac
}

# GPU lock-based queue
LOCK_DIR=$(mktemp -d)
trap 'rm -rf "$LOCK_DIR"' EXIT

acquire_gpu() {
    while true; do
        for gpu in "${GPUS[@]}"; do
            if mkdir "$LOCK_DIR/lock_$gpu" 2>/dev/null; then
                echo "$gpu"
                return
            fi
        done
        sleep 1
    done
}
release_gpu() { rmdir "$LOCK_DIR/lock_$1" 2>/dev/null || true; }

run_gpu_job() {
    local model=$1 dataset=$2 split=$3 attack=$4
    local outfile="$OUT_F/${model}_${attack}_5k_${dataset}_${split}.npz"
    if [ -f "$outfile" ]; then
        echo "[$(date +%H:%M:%S)] SKIP: $model $attack $dataset $split"
        return 0
    fi
    local gpu; gpu=$(acquire_gpu)
    local logfile="$LOG_DIR/${model}_${dataset}_${split}_${attack}.log"
    local overrides; overrides=$(dataset_overrides "$dataset")
    echo "[$(date +%H:%M:%S)] GPU $gpu: $model $attack $dataset $split"
    CUDA_VISIBLE_DEVICES=$gpu \
        "$P" -u "$CDI_DIR/main.py" \
        +model=$model +action=features_extraction +attack=$attack \
        +dataset=$dataset dataset.split=$split \
        ++cfg.path_to_features="$OUT_F" ++cfg.path_to_scores="$OUT_S" \
        ++action.device=cuda:0 \
        ${overrides} \
        > "$logfile" 2>&1
    local rc=$?
    release_gpu "$gpu"
    if [ $rc -ne 0 ]; then
        echo "  FAILED: $model $attack $dataset $split (see $logfile)"
    fi
    return $rc
}

run_cpu_job() {
    local model=$1 action=$2 attack=$3 dataset=$4 split=$5
    local outdir; if [ "$action" = "scores_computation" ]; then outdir="$OUT_S"; else outdir="$OUT_F"; fi
    local outfile="$outdir/${model}_${attack}_5k_${dataset}_${split}.npz"
    if [ -f "$outfile" ]; then
        echo "[$(date +%H:%M:%S)] SKIP (cpu): $model $action $attack $dataset $split"
        return 0
    fi
    local logfile="$LOG_DIR/${model}_${dataset}_${split}_${action}_${attack}.log"
    local overrides; overrides=$(dataset_overrides "$dataset")
    echo "[$(date +%H:%M:%S)] CPU $action: $model $attack $dataset $split"
    CUDA_VISIBLE_DEVICES="" \
        "$P" -u "$CDI_DIR/main.py" \
        +model=$model +action=$action +attack=$attack \
        +dataset=$dataset dataset.split=$split \
        ++cfg.path_to_features="$OUT_F" ++cfg.path_to_scores="$OUT_S" \
        action.device=cpu \
        ${overrides} \
        > "$logfile" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "  FAILED: $action $attack $dataset $split (see $logfile)"
    fi
    return $rc
}

echo "============================================"
echo "MODELS:    ${MODELS[*]}"
echo "DATASETS:  ${DATASETS[*]}"
echo "SPLITS:    ${SPLITS[*]}"
echo "ATTACKS:   ${GPU_ATTACKS[*]}"
echo "GPUS:      ${GPUS[*]}"
echo "OUT_ROOT:  $OUT_ROOT"
[ -n "${IMAGENET_PATH:-}" ] && echo "IMAGENET_PATH: $IMAGENET_PATH"
echo "============================================"
START=$(date +%s)

for model in "${MODELS[@]}"; do
    echo "================= MODEL: $model ================="

    # Phase 1 — parallel GPU feature extraction (lock-based queue across all GPUs)
    echo "--- Phase 1: GPU features ---"
    PIDS=()
    for dataset in "${DATASETS[@]}"; do
        for split in "${SPLITS[@]}"; do
            for attack in "${GPU_ATTACKS[@]}"; do
                run_gpu_job "$model" "$dataset" "$split" "$attack" &
                PIDS+=($!)
            done
        done
    done
    FAIL=0
    for pid in "${PIDS[@]}"; do
        wait "$pid" || FAIL=$((FAIL + 1))
    done
    echo "Phase 1 complete ($FAIL failures)"

    # Phase 2 — CPU carlini_lt scores (input to combination_attack)
    echo "--- Phase 2: carlini_lt scores ---"
    for dataset in "${DATASETS[@]}"; do
        for split in "${SPLITS[@]}"; do
            run_cpu_job "$model" scores_computation carlini_lt "$dataset" "$split"
        done
    done

    # Phase 3 — CPU combination_attack features
    echo "--- Phase 3: combination_attack features ---"
    for dataset in "${DATASETS[@]}"; do
        for split in "${SPLITS[@]}"; do
            run_cpu_job "$model" features_extraction combination_attack "$dataset" "$split"
        done
    done

    # Phase 4 — CPU combination_attack scores
    echo "--- Phase 4: combination_attack scores ---"
    for dataset in "${DATASETS[@]}"; do
        for split in "${SPLITS[@]}"; do
            run_cpu_job "$model" scores_computation combination_attack "$dataset" "$split"
        done
    done

    echo "DONE model=$model"
done

TOTAL=$(( $(date +%s) - START ))
echo "============================================"
echo "ALL DONE in ${TOTAL}s"
echo "Features: $(ls "$OUT_F" | wc -l) files"
echo "Scores:   $(ls "$OUT_S" | wc -l) files"
echo "============================================"
