#!/usr/bin/env python3
"""
PCA visualization of combination_attack features.

For each generator setting (e.g. var_last4), plots 4 point clouds:
  - TP train: autoencoded ImageNet-train (real member data)
  - TP val:   img2img ImageNet-train (synthetic from members)
  - FP train: autoencoded ImageNet-val (real non-member data)
  - FP val:   img2img ImageNet-val (synthetic from non-members)

If the synthetic generation is faithful, TP-train should separate
from the other three, which should overlap heavily.
"""
import argparse
import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

MODEL = "dit_rf"
RUN_ID = "5k"

# out/ (default): VAR and Infinity with _novqvae suffix not in filenames
SETTINGS_OUT = {
    "var_last2": "VAR (last 2 / 10 scales generated)",
    "var_last4": "VAR (last 4 / 10 scales generated)",
    "var_last6": "VAR (last 6 / 10 scales generated)",
    "var_last8": "VAR (last 8 / 10 scales generated)",
    "infinity_last1": "Infinity (last 1 / 7 scales generated)",
    "infinity_last2": "Infinity (last 2 / 7 scales generated)",
    "infinity_last3": "Infinity (last 3 / 7 scales generated)",
    "infinity_last4": "Infinity (last 4 / 7 scales generated)",
}

# out_novqvae: filenames use e.g. tp_var_last4_novqvae_train
SETTINGS_NOVQVAE = {
    "var_last4_novqvae": "VAR no-VQVAE (last 4 / 10 scales)",
    "var_last6_novqvae": "VAR no-VQVAE (last 6 / 10 scales)",
    "var_last10_novqvae": "VAR no-VQVAE (last 10 / 10 scales)",
    "infinity_last4_novqvae": "Infinity no-VQVAE (last 4 / 7 scales)",
}

# out_real: single dataset, train vs val (sanity check: members vs non-members)
SETTINGS_REAL = {"imagenet_5pc": "Real ImageNet 5%"}
GROUPS_REAL = [
    ("imagenet_5pc", "train", "Train (member)", "tab:red", "o"),
    ("imagenet_5pc", "val", "Val (non-member)", "tab:blue", "s"),
]

GROUPS = [
    ("tp", "train", "TP train — autoencoded member data (real)", "tab:red", "o"),
    ("tp", "val", "TP val — img2img member data (synthetic)", "tab:orange", "^"),
    ("fp", "train", "FP train — autoencoded non-member data (real)", "tab:blue", "s"),
    ("fp", "val", "FP val — img2img non-member data (synthetic)", "tab:cyan", "D"),
]


def load_features(prefix, split):
    name = f"{MODEL}_combination_attack_{RUN_ID}_{prefix}_{split}.npz"
    path = os.path.join(FEAT_DIR, name)
    f = np.load(path, allow_pickle=True)
    data = f["data"].astype(np.float32).reshape(f["data"].shape[0], -1)
    return data


def plot_setting(setting_key, setting_label):
    arrays, labels, colors, markers = [], [], [], []
    for prefix_type, split, label, color, marker in GROUPS:
        key = f"{prefix_type}_{setting_key}"
        try:
            arr = load_features(key, split)
        except FileNotFoundError:
            print(f"  missing: {key} {split}, skipping")
            continue
        arrays.append(arr)
        labels.append(label)
        colors.append(color)
        markers.append(marker)

    if len(arrays) < 2:
        return

    combined = np.concatenate(arrays, axis=0)
    pca = PCA(n_components=2, random_state=42)
    projected = pca.fit_transform(combined)

    fig, ax = plt.subplots(figsize=(9, 7))
    offset = 0
    for arr, label, color, marker in zip(arrays, labels, colors, markers):
        n = arr.shape[0]
        pts = projected[offset : offset + n]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=color,
            marker=marker,
            s=8,
            alpha=0.35,
            label=label,
            edgecolors="none",
        )
        offset += n

    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%} variance)")
    ax.set_title(f"Combination-attack features — {setting_label}")
    ax.legend(loc="best", fontsize=8, markerscale=2.5, framealpha=0.9)
    fig.tight_layout()

    fname = f"pca_{setting_key}.png"
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=200)
    plt.close(fig)
    print(f"  saved {fname}")


def plot_setting_real(groups_list, setting_label):
    """Single plot with N groups (e.g. train vs val for real ImageNet)."""
    arrays, labels, colors, markers = [], [], [], []
    for prefix, split, label, color, marker in groups_list:
        try:
            arr = load_features(prefix, split)
        except FileNotFoundError:
            print(f"  missing: {prefix} {split}, skipping")
            continue
        arrays.append(arr)
        labels.append(label)
        colors.append(color)
        markers.append(marker)

    if len(arrays) < 2:
        return

    combined = np.concatenate(arrays, axis=0)
    pca = PCA(n_components=2, random_state=42)
    projected = pca.fit_transform(combined)

    fig, ax = plt.subplots(figsize=(9, 7))
    offset = 0
    for arr, label, color, marker in zip(arrays, labels, colors, markers):
        n = arr.shape[0]
        pts = projected[offset : offset + n]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=color,
            marker=marker,
            s=8,
            alpha=0.35,
            label=label,
            edgecolors="none",
        )
        offset += n

    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%} variance)")
    ax.set_title(f"Combination-attack features — {setting_label}")
    ax.legend(loc="best", fontsize=8, markerscale=2.5, framealpha=0.9)
    fig.tight_layout()

    fname = "pca_imagenet_5pc_real.png"
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=200)
    plt.close(fig)
    print(f"  saved {fname}")


def main():
    parser = argparse.ArgumentParser(description="PCA of combination_attack features")
    parser.add_argument(
        "--out_dir",
        default="out",
        choices=("out", "out_novqvae", "out_real"),
        help="Base dir: out, out_novqvae, or out_real (default: out)",
    )
    args = parser.parse_args()
    base = os.path.join(os.path.dirname(__file__), args.out_dir)
    feat_dir = os.path.join(base, "features")
    out_dir = os.path.join(base, "plots")

    global FEAT_DIR, OUT_DIR
    FEAT_DIR = feat_dir
    OUT_DIR = out_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.out_dir == "out_real":
        for key, label in SETTINGS_REAL.items():
            print(f"[{key}]")
            plot_setting_real(GROUPS_REAL, label)
    else:
        settings = SETTINGS_NOVQVAE if args.out_dir == "out_novqvae" else SETTINGS_OUT
        for key, label in settings.items():
            print(f"[{key}]")
            plot_setting(key, label)

    print(f"\nAll plots in {OUT_DIR}")


if __name__ == "__main__":
    main()
