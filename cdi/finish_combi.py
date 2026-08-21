#!/usr/bin/env python3
"""
Standalone script to compute carlini_lt scores and combination_attack features
without loading the diffusion model. Runs on CPU alongside GPU feature extraction.

Monitors the features directory and processes datasets as their prerequisites
become available.
"""
import os
import time
import numpy as np
import torch

FEAT_DIR = "./out_novqvae/features"
SCORE_DIR = "./out_novqvae/scores"
MODEL = "dit_rf"
RUN_ID = "5k"
TOTAL_SAMPLES = 5000  # n_samples_eval (2500) + train_samples (2500) + valid_samples (0)

DATASETS = [
    "tp_var_last4_novqvae",
    "tp_var_last6_novqvae",
    "tp_var_last10_novqvae",
    "fp_var_last4_novqvae",
    "fp_var_last6_novqvae",
    "fp_var_last10_novqvae",
    "tp_infinity_last4_novqvae",
    "tp_infinity_last7_novqvae",
    "fp_infinity_last4_novqvae",
    "fp_infinity_last7_novqvae",
]
SPLITS = ["train", "val"]
BASE_ATTACKS = ["carlini_lt", "gradient_masking", "multiple_loss", "clid"]
SCORE_ATTACKS = ["carlini_lt"]
FEATURE_ATTACKS = ["gradient_masking", "multiple_loss", "clid"]


def fname(attack, ds, split, directory):
    return os.path.join(directory, f"{MODEL}_{attack}_{RUN_ID}_{ds}_{split}.npz")


def exists(attack, ds, split, directory):
    return os.path.isfile(fname(attack, ds, split, directory))


def load_npz(path):
    f = np.load(path, allow_pickle=True)
    return f["data"], f["metadata"][()]


def save_npz(path, data, metadata):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, data=data, metadata=metadata)


def compute_carlini_scores(ds, split):
    """Carlini score = mean over measurements dimension. Input shape: (N, M, 1) -> (N,)"""
    out_path = fname("carlini_lt", ds, split, SCORE_DIR)
    if os.path.isfile(out_path):
        return True

    feat_path = fname("carlini_lt", ds, split, FEAT_DIR)
    if not os.path.isfile(feat_path):
        return False

    data, metadata = load_npz(feat_path)
    t = torch.from_numpy(data)
    scores = t.mean(dim=1).squeeze(1).numpy()
    assert scores.shape[0] == TOTAL_SAMPLES, f"Expected {TOTAL_SAMPLES}, got {scores.shape[0]}"

    save_npz(out_path, scores, metadata)
    print(f"  [scores] carlini_lt {ds} {split} -> {out_path}")
    return True


def compute_combination_features(ds, split):
    """Concatenate carlini_lt scores + gradient_masking/multiple_loss/clid features."""
    out_path = fname("combination_attack", ds, split, FEAT_DIR)
    if os.path.isfile(out_path):
        return True

    for attack in SCORE_ATTACKS:
        if not exists(attack, ds, split, SCORE_DIR):
            return False
    for attack in FEATURE_ATTACKS:
        if not exists(attack, ds, split, FEAT_DIR):
            return False

    parts = []
    for attack in SCORE_ATTACKS:
        data, _ = load_npz(fname(attack, ds, split, SCORE_DIR))
        t = torch.from_numpy(data)
        parts.append(t.view(-1, 1, 1))

    metadata = None
    for attack in FEATURE_ATTACKS:
        data, meta = load_npz(fname(attack, ds, split, FEAT_DIR))
        if metadata is None:
            metadata = meta
        t = torch.from_numpy(data)
        B = t.shape[0]
        parts.append(t.view(B, 1, -1))

    combined = torch.cat(parts, dim=2).numpy()  # (B, 1, N_features)
    assert combined.shape[0] == TOTAL_SAMPLES, f"Expected {TOTAL_SAMPLES}, got {combined.shape[0]}"

    save_npz(out_path, combined, metadata)
    print(f"  [combo]  combination_attack {ds} {split} -> {out_path}")
    return True


def main():
    print("=" * 60)
    print("Combination attack feature extraction (CPU-only)")
    print("=" * 60)

    max_retries = 120  # 120 * 30s = 1 hour max wait
    retry = 0

    while retry < max_retries:
        all_done = True
        progress_made = False

        # Phase 1: carlini_lt scores
        for ds in DATASETS:
            for split in SPLITS:
                if not exists("carlini_lt", ds, split, SCORE_DIR):
                    ok = compute_carlini_scores(ds, split)
                    if ok:
                        progress_made = True
                    else:
                        all_done = False

        # Phase 2: combination_attack features
        for ds in DATASETS:
            for split in SPLITS:
                if not exists("combination_attack", ds, split, FEAT_DIR):
                    ok = compute_combination_features(ds, split)
                    if ok:
                        progress_made = True
                    else:
                        all_done = False

        if all_done:
            print("\nAll combination_attack features extracted!")
            break

        if progress_made:
            retry = 0
        else:
            retry += 1

        pending_scores = sum(
            1 for ds in DATASETS for sp in SPLITS
            if not exists("carlini_lt", ds, sp, SCORE_DIR)
        )
        pending_combo = sum(
            1 for ds in DATASETS for sp in SPLITS
            if not exists("combination_attack", ds, sp, FEAT_DIR)
        )
        print(f"\n[wait] {pending_scores} carlini scores + {pending_combo} combo features pending. "
              f"Waiting 30s for GPU jobs to finish base features... (attempt {retry}/{max_retries})")
        time.sleep(30)

    # Final summary
    print("\n" + "=" * 60)
    print("Final status:")
    for ds in DATASETS:
        for split in SPLITS:
            score_ok = "OK" if exists("carlini_lt", ds, split, SCORE_DIR) else "MISSING"
            combo_ok = "OK" if exists("combination_attack", ds, split, FEAT_DIR) else "MISSING"
            base_status = []
            for a in BASE_ATTACKS:
                if exists(a, ds, split, FEAT_DIR):
                    base_status.append(a[:4])
                else:
                    base_status.append("----")
            print(f"  {ds:35s} {split:5s}  base=[{','.join(base_status)}]  score={score_ok:7s}  combo={combo_ok}")
    print("=" * 60)


if __name__ == "__main__":
    main()
