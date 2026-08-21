#!/bin/bash
# Stage extracted strength-ablation .npz features into the layout that
# duci-pul/main.py + src/data.py expect:
#
#   data/<target>/<target>_real_mem.npz       <- imagenet_5pc/train (real members)
#   data/<target>/<target>_real_nonmem.npz    <- imagenet_5pc/val   (real non-members)
#   data/<target>/<target>_ae_mem.npz         <- tp_sd_strength*/train  (AE members)
#   data/<target>/<target>_ae_nonmem.npz      <- fp_sd_strength*/train  (AE non-members)
#   data/<target>/<target>_from-mem.npz       <- tp_sd_strength*/val    (img2img members)
#   data/<target>/<target>_from-nonmem.npz    <- fp_sd_strength*/val    (img2img non-members)
#
# We use symlinks to avoid duplicating the npz files.
set -e
ROOT=/home/jdubinsk/dataset_inference_dm/duci-pul
SRC=$ROOT/iars/out_strength_ablation/features
DST_BASE=$ROOT/data
ATTACK=llm_mia_cfg
RUN_ID=5k

stage_one() {
  local model="$1" strength="$2"
  local target="${model}_s${strength}"
  local dst="$DST_BASE/$target"
  mkdir -p "$dst"
  ln -sf "$SRC/${model}_${ATTACK}_${RUN_ID}_imagenet_5pc_train.npz"        "$dst/${target}_real_mem.npz"
  ln -sf "$SRC/${model}_${ATTACK}_${RUN_ID}_imagenet_5pc_val.npz"          "$dst/${target}_real_nonmem.npz"
  ln -sf "$SRC/${model}_${ATTACK}_${RUN_ID}_tp_sd_strength${strength}_train.npz" "$dst/${target}_ae_mem.npz"
  ln -sf "$SRC/${model}_${ATTACK}_${RUN_ID}_fp_sd_strength${strength}_train.npz" "$dst/${target}_ae_nonmem.npz"
  ln -sf "$SRC/${model}_${ATTACK}_${RUN_ID}_tp_sd_strength${strength}_val.npz"   "$dst/${target}_from-mem.npz"
  ln -sf "$SRC/${model}_${ATTACK}_${RUN_ID}_fp_sd_strength${strength}_val.npz"   "$dst/${target}_from-nonmem.npz"
  echo "[staged] $dst"
  ls -l "$dst" | awk '{print "  ", $9, "->", $11}'
}

for model in var_24 rar_xl; do
  for s in 0.4 0.6; do
    stage_one "$model" "$s"
  done
done
