#!/bin/bash

CDI_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$CDI_DIR/../.venv/bin/python"

DATASETS=(
    tp_var_last2 tp_var_last4 tp_var_last6 tp_var_last8
    fp_var_last2 fp_var_last4 fp_var_last6 fp_var_last8
    tp_infinity_last1 tp_infinity_last2 tp_infinity_last3 tp_infinity_last4
    fp_infinity_last1 fp_infinity_last2 fp_infinity_last3 fp_infinity_last4
)

GPU_ATTACKS=(carlini_lt gradient_masking multiple_loss clid)
SPLITS=(train val)
GPUS=(0 1 2)
NUM_GPUS=${#GPUS[@]}

MODEL=dit_rf
LOG_DIR="$CDI_DIR/logs"
mkdir -p "$LOG_DIR"

export PYTHONDONTWRITEBYTECODE=1

# ── GPU job queue using lock files ──────────────────────────────────
LOCK_DIR=$(mktemp -d)
for gpu in "${GPUS[@]}"; do
    touch "$LOCK_DIR/gpu_$gpu"
done

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

release_gpu() {
    rmdir "$LOCK_DIR/lock_$1" 2>/dev/null
}

run_gpu_job() {
    local dataset=$1 split=$2 attack=$3
    local gpu
    gpu=$(acquire_gpu)
    local logfile="$LOG_DIR/${dataset}_${split}_${attack}.log"
    local outfile="$CDI_DIR/out/features/${MODEL}_${attack}_5k_${dataset}_${split}.npz"
    if [ -f "$outfile" ]; then
        release_gpu "$gpu"
        echo "[$(date +%H:%M:%S)] SKIP (exists): $attack $dataset $split"
        return 0
    fi
    echo "[$(date +%H:%M:%S)] GPU $gpu: $attack $dataset $split"

    CUDA_VISIBLE_DEVICES=$gpu \
        "$PYTHON" -u "$CDI_DIR/main.py" \
        +model=$MODEL +action=features_extraction +attack=$attack \
        +dataset=$dataset dataset.split=$split \
        > "$logfile" 2>&1
    local rc=$?

    release_gpu "$gpu"
    if [ $rc -ne 0 ]; then
        echo "  FAILED: $attack $dataset $split (see $logfile)"
    fi
    return $rc
}

run_cpu() {
    local action=$1 attack=$2 dataset=$3 split=$4
    local logfile="$LOG_DIR/${dataset}_${split}_${attack}_${action}.log"
    echo "[$(date +%H:%M:%S)] CPU $action: $attack $dataset $split"
    CUDA_VISIBLE_DEVICES="" \
        "$PYTHON" -u "$CDI_DIR/main.py" \
        +model=$MODEL +action=$action +attack=$attack \
        +dataset=$dataset dataset.split=$split \
        action.device=cpu \
        > "$logfile" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "  FAILED: $action $attack $dataset $split (see $logfile)"
    fi
    return $rc
}

echo "============================================"
echo "PHASE 1: GPU Feature Extraction"
echo "  128 jobs across ${NUM_GPUS} GPUs (lock-based queue)"
echo "============================================"
START=$(date +%s)

PIDS=()
for dataset in "${DATASETS[@]}"; do
    for split in "${SPLITS[@]}"; do
        for attack in "${GPU_ATTACKS[@]}"; do
            run_gpu_job "$dataset" "$split" "$attack" &
            PIDS+=($!)
        done
    done
done

FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=$((FAIL + 1))
done

ELAPSED=$(( $(date +%s) - START ))
echo "Phase 1 complete in ${ELAPSED}s ($FAIL failures). Features: $(ls "$CDI_DIR/out/features/" | grep -v "imagenet_real\|mscoco" | wc -l)"
rm -rf "$LOCK_DIR"

if [ $FAIL -gt 0 ]; then
    echo "WARNING: $FAIL jobs failed. Check logs in $LOG_DIR"
fi

echo "============================================"
echo "PHASE 2: Carlini Scores (CPU)"
echo "============================================"

for dataset in "${DATASETS[@]}"; do
    for split in "${SPLITS[@]}"; do
        run_cpu scores_computation carlini_lt "$dataset" "$split"
    done
done
echo "Phase 2 complete."

echo "============================================"
echo "PHASE 3: Combination Attack Features (CPU)"
echo "============================================"

for dataset in "${DATASETS[@]}"; do
    for split in "${SPLITS[@]}"; do
        run_cpu features_extraction combination_attack "$dataset" "$split"
    done
done
echo "Phase 3 complete."

echo "============================================"
echo "PHASE 4: Combination Attack Scores (CPU)"
echo "============================================"

for dataset in "${DATASETS[@]}"; do
    for split in "${SPLITS[@]}"; do
        run_cpu scores_computation combination_attack "$dataset" "$split"
    done
done
echo "Phase 4 complete."

TOTAL=$(( $(date +%s) - START ))
echo "============================================"
echo "ALL DONE in ${TOTAL}s"
echo "============================================"
echo "Features: $(ls "$CDI_DIR/out/features/" | grep -v "imagenet_real\|mscoco" | wc -l) files"
echo "Scores:   $(ls "$CDI_DIR/out/scores/" | wc -l) files"
