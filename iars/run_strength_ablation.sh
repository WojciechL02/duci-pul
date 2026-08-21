#!/bin/bash
# Strength ablation for VAR-24 and RAR-XL.
#
# Extracts the combination-attack (`llm_mia_cfg`) feature suite for
#   models    : var_24, rar_xl
#   reference : real ImageNet-5pc + tp/fp at SD img2img strength 0.4 and 0.6
#                (strength 0.5 is already covered in the paper main table)
#   splits    : train, val (per dataset, mirrors the same-family ablation)
#
# Each (model, dataset, split) job is dispatched to one of the 3 local GPUs.
# Idempotent: skips runs whose .npz output already exists.

set -e
cd /home/jdubinsk/dataset_inference_dm/duci-pul/iars

P=/home/jdubinsk/dataset_inference_dm/duci-pul/.venv/bin/python3
ATTACK="${ATTACK:-llm_mia_cfg}"
RUN_ID="${RUN_ID:-5k}"
OUT_F="$(pwd)/out_strength_ablation/features"
OUT_S="$(pwd)/out_strength_ablation/scores"
LOG_DIR="$(pwd)/out_strength_ablation/logs"
mkdir -p "$OUT_F" "$OUT_S" "$LOG_DIR"

COMMON="++cfg.path_to_features=$OUT_F ++cfg.path_to_scores=$OUT_S ++cfg.run_id=$RUN_ID"

MODELS=(var_24 rar_xl)
DATASETS=(
  imagenet_5pc
  fp_sd_strength0.4
  tp_sd_strength0.4
  fp_sd_strength0.6
  tp_sd_strength0.6
)
SPLITS=(train val)

# Build the full job list as "model|dataset|split" strings.
JOBS=()
for MODEL in "${MODELS[@]}"; do
  for DS in "${DATASETS[@]}"; do
    for SPLIT in "${SPLITS[@]}"; do
      OUT="$OUT_F/${MODEL}_${ATTACK}_${RUN_ID}_${DS}_${SPLIT}.npz"
      if [ -f "$OUT" ]; then
        echo "[skip] $OUT"
        continue
      fi
      JOBS+=("${MODEL}|${DS}|${SPLIT}")
    done
  done
done

NJOBS=${#JOBS[@]}
echo "=== Strength ablation: $NJOBS jobs to run ==="
if [ "$NJOBS" -eq 0 ]; then
  echo "Nothing to do."
  exit 0
fi

run_one() {
  local gpu="$1" job="$2"
  local model ds split
  IFS='|' read -r model ds split <<< "$job"
  local logf="$LOG_DIR/${model}_${ATTACK}_${RUN_ID}_${ds}_${split}.log"
  echo "[run]  GPU $gpu | $model | $ds | $split  -> $logf"
  CUDA_VISIBLE_DEVICES="$gpu" \
    "$P" main.py +model="$model" +attack="$ATTACK" +dataset="$ds" \
      ++dataset.split="$split" $COMMON ++cfg.device=cuda:0 \
      > "$logf" 2>&1
}

# Dispatch jobs in waves of 3 (one per GPU).
NGPU=3
i=0
while [ "$i" -lt "$NJOBS" ]; do
  pids=()
  for ((g=0; g<NGPU && i<NJOBS; g++)); do
    run_one "$g" "${JOBS[$i]}" &
    pids+=("$!")
    i=$((i+1))
  done
  # Wait for this wave to finish before starting the next.
  for pid in "${pids[@]}"; do
    wait "$pid" || echo "[warn] job (pid=$pid) failed"
  done
done

echo "=== STRENGTH ABLATION DONE ==="
ls -la "$OUT_F" | tail -40
