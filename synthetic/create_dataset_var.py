#!/usr/bin/env python3
"""
Create synthetic ImageNet datasets using VAR for dataset inference.

Produces:
  imagenet_tp_var_nreal{N}_cfg{C}/
    train/  - IMAGENET TRAIN autoencoded through VQVAE
    val/    - IMAGENET TRAIN img2img (first N scales real, last K-N generated)
    metadata.md

  imagenet_fp_var_nreal{N}_cfg{C}/
    train/  - IMAGENET VAL autoencoded through VQVAE
    val/    - IMAGENET VAL img2img
    metadata.md

VAR uses 10 scales: patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16).
n_real_scales=8 keeps the first 8 scales real, generates last 2.
"""
import argparse
import os
import sys
import time
import datetime
from concurrent.futures import ThreadPoolExecutor

# Must set CUDA_VISIBLE_DEVICES before importing torch
_gpu_arg = None
for i, a in enumerate(sys.argv):
    if a == "--gpu" and i + 1 < len(sys.argv):
        _gpu_arg = sys.argv[i + 1]
if _gpu_arg is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_arg

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "VAR"))

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import dist  # noqa: F401  (VAR internals may reference this)
from models import build_vae_var
from models.helpers import sample_with_top_k_top_p_

WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights", "VAR")
VAE_CKPT = os.path.join(WEIGHTS_DIR, "vae_ch160v4096z32.pth")
VAR_CKPT = os.path.join(WEIGHTS_DIR, "var_d30.pth")
PATCH_NUMS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ImageNetSubset(Dataset):
    """ImageNet split that returns (image_tensor, class_idx, synset_name, filename)."""

    def __init__(self, root, tf, max_images=None):
        self.root = root
        self.tf = tf
        self.samples = []

        synsets = sorted(
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d)) and d.startswith("n")
        )
        self.class_to_idx = {s: i for i, s in enumerate(synsets)}

        for syn in synsets:
            syn_dir = os.path.join(root, syn)
            for fname in sorted(os.listdir(syn_dir)):
                if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    self.samples.append((
                        os.path.join(syn_dir, fname),
                        self.class_to_idx[syn],
                        syn,
                        fname,
                    ))
            if max_images and len(self.samples) >= max_images:
                self.samples = self.samples[:max_images]
                break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, syn, fname = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.tf(img), label, syn, fname


def _collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    syns = [b[2] for b in batch]
    fnames = [b[3] for b in batch]
    return imgs, labels, syns, fnames


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_models(device):
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=PATCH_NUMS,
        num_classes=1000, depth=30, shared_aln=False,
        attn_l2_norm=True,
    )
    vae.load_state_dict(torch.load(VAE_CKPT, map_location="cpu"), strict=True)
    var.load_state_dict(torch.load(VAR_CKPT, map_location="cpu"), strict=True)
    vae.eval()
    var.eval()
    return vae, var


# ---------------------------------------------------------------------------
# Core operations (batched)
# ---------------------------------------------------------------------------
@torch.no_grad()
def batch_autoencoder(vae, imgs):
    """VQVAE reconstruction.  imgs: [B,3,256,256] in [-1,1] -> [0,1]."""
    return vae.img_to_reconstructed_img(imgs, last_one=True).add_(1).mul_(0.5)


@torch.no_grad()
def batch_img2img(var_model, vae, imgs, labels, n_real_scales,
                  cfg, top_k, top_p, seed):
    """Batched img2img.  imgs: [B,3,256,256] in [-1,1], labels: [B] -> [0,1]."""
    device = imgs.device
    B = imgs.shape[0]
    patch_nums = var_model.patch_nums

    var_model.rng.manual_seed(seed)
    rng = var_model.rng

    real_idx_Bls = vae.img_to_idxBl(imgs)

    label_B = labels.to(device)
    sos = cond_BD = var_model.class_emb(
        torch.cat((label_B,
                    torch.full_like(label_B, fill_value=var_model.num_classes)),
                   dim=0)
    )

    lvl_pos = var_model.lvl_embed(var_model.lvl_1L) + var_model.pos_1LC
    next_token_map = (
        sos.unsqueeze(1).expand(2 * B, var_model.first_l, -1)
        + var_model.pos_start.expand(2 * B, var_model.first_l, -1)
        + lvl_pos[:, :var_model.first_l]
    )

    cur_L = 0
    f_hat = sos.new_zeros(B, var_model.Cvae, patch_nums[-1], patch_nums[-1])

    for b in var_model.blocks:
        b.attn.kv_caching(True)

    for si, pn in enumerate(patch_nums):
        ratio = si / var_model.num_stages_minus_1
        cur_L += pn * pn

        x = next_token_map
        cond_BD_or_gss = var_model.shared_ada_lin(cond_BD)
        for b in var_model.blocks:
            x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
        logits_BlV = var_model.get_logits(x, cond_BD)

        t = cfg * ratio
        logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]

        if si < n_real_scales:
            idx_Bl = real_idx_Bls[si].to(device)
        else:
            idx_Bl = sample_with_top_k_top_p_(
                logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1
            )[:, :, 0]

        h_BChw = var_model.vae_quant_proxy[0].embedding(idx_Bl)
        h_BChw = h_BChw.transpose_(1, 2).reshape(B, var_model.Cvae, pn, pn)
        f_hat, next_token_map = \
            var_model.vae_quant_proxy[0].get_next_autoregressive_input(
                si, len(patch_nums), f_hat, h_BChw)
        if si != var_model.num_stages_minus_1:
            next_token_map = next_token_map.view(
                B, var_model.Cvae, -1).transpose(1, 2)
            next_token_map = (
                var_model.word_embed(next_token_map)
                + lvl_pos[:, cur_L:cur_L + patch_nums[si + 1] ** 2]
            )
            next_token_map = next_token_map.repeat(2, 1, 1)

    for b in var_model.blocks:
        b.attn.kv_caching(False)

    return var_model.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def _save_one(img_np, path):
    Image.fromarray(img_np).save(path)


def save_batch(tensor_01, syns, fnames, out_root, executor):
    """Async-save a [B,3,H,W] float-[0,1] batch as PNGs."""
    arr = (tensor_01.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    futs = []
    for i in range(arr.shape[0]):
        d = os.path.join(out_root, syns[i])
        os.makedirs(d, exist_ok=True)
        stem = os.path.splitext(fnames[i])[0]
        futs.append(executor.submit(_save_one, arr[i], os.path.join(d, stem + ".png")))
    return futs


def write_metadata(out_dir, args, split_name, n_imgs, elapsed):
    with open(os.path.join(out_dir, "metadata.md"), "w") as f:
        f.write(f"# {os.path.basename(out_dir)}\n\n")
        f.write(f"**Source split**: `{split_name}` from `{args.imagenet_dir}`\n\n")
        f.write("## Model\n\n")
        f.write("- VAR-d30 (2 B params) + VQVAE (ch160, V4096, z32)\n\n")
        f.write("## Hyperparameters\n\n")
        f.write(f"| param | value |\n|---|---|\n")
        f.write(f"| last_scales | {args.last_scales} (generate last "
                f"{args.last_scales} of 10) |\n")
        f.write(f"| n_real_scales | {args.n_real_scales} (scales 0–"
                f"{args.n_real_scales - 1} from real image) |\n")
        f.write(f"| cfg | {args.cfg} |\n")
        f.write(f"| top_k | {args.top_k} |\n")
        f.write(f"| top_p | {args.top_p} |\n")
        f.write(f"| seed | {args.seed} |\n")
        f.write(f"| batch_size | {args.batch_size} |\n\n")
        f.write("## Directories\n\n")
        f.write("- `train/` — VQVAE autoencoder reconstruction\n")
        f.write(f"- `val/` — img2img (first {args.n_real_scales} scales real, "
                f"last {10 - args.n_real_scales} generated)\n\n")
        f.write(f"## Stats\n\n- images: {n_imgs}\n- time: {elapsed:.1f} s\n"
                f"- created: {datetime.datetime.now().isoformat()}\n")


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------
def process_split(vae, var, split_dir, out_dir, args, split_name):
    print(f"\n{'=' * 60}")
    print(f"Processing {split_name}: {split_dir} -> {out_dir}")
    print(f"{'=' * 60}")

    tf = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    ds = ImageNetSubset(split_dir, tf, max_images=args.max_images)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=min(4, args.batch_size), pin_memory=True,
        collate_fn=_collate,
    )

    device = next(var.parameters()).device
    train_dir = os.path.join(out_dir, "train")
    val_dir = os.path.join(out_dir, "val")
    executor = ThreadPoolExecutor(max_workers=8)
    all_futs = []

    total = len(ds)
    done = 0
    t0 = time.time()

    for bi, (imgs, labels, syns, fnames) in enumerate(loader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        B = imgs.shape[0]

        recon = batch_autoencoder(vae, imgs)
        all_futs += save_batch(recon, syns, fnames, train_dir, executor)

        synth = batch_img2img(
            var, vae, imgs, labels,
            n_real_scales=args.n_real_scales,
            cfg=args.cfg, top_k=args.top_k, top_p=args.top_p,
            seed=args.seed + bi,
        )
        all_futs += save_batch(synth, syns, fnames, val_dir, executor)

        done += B
        rate = done / (time.time() - t0)
        print(f"  [{done}/{total}] {rate:.1f} img/s")

    for fu in all_futs:
        fu.result()

    elapsed = time.time() - t0
    write_metadata(out_dir, args, split_name, total, elapsed)
    print(f"Finished {split_name}: {total} images in {elapsed:.1f} s "
          f"({total / max(elapsed, 1e-6):.1f} img/s)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    pa = argparse.ArgumentParser(
        description="Create VAR synthetic ImageNet datasets")
    pa.add_argument("--imagenet_dir", type=str,
                    default=os.path.join(SCRIPT_DIR, "..", "cdi", "data", "imagenet_5pc"))
    pa.add_argument("--output_root", type=str, default=os.path.join(SCRIPT_DIR, "datasets"))
    pa.add_argument("--last_scales", type=int, required=True,
                    help="Number of scales to generate (last N of 10)")
    pa.add_argument("--cfg", type=float, default=1.5)
    pa.add_argument("--top_k", type=int, default=900)
    pa.add_argument("--top_p", type=float, default=0.96)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--batch_size", type=int, default=16)
    pa.add_argument("--max_images", type=int, default=None)
    pa.add_argument("--gpu", type=int, default=1)
    pa.add_argument("--skip_tp", action="store_true")
    pa.add_argument("--skip_fp", action="store_true")
    args = pa.parse_args()

    assert 0 < args.last_scales <= 10, "last_scales must be 1–10"
    args.n_real_scales = 10 - args.last_scales

    device = "cuda"

    tag = f"var_last{args.last_scales}_cfg{args.cfg}"
    tp_dir = os.path.join(args.output_root, f"imagenet_tp_{tag}")
    fp_dir = os.path.join(args.output_root, f"imagenet_fp_{tag}")

    print("Loading VAR-d30 + VQVAE …")
    vae, var = load_models(device)
    print("Models ready.")

    if not args.skip_tp:
        process_split(vae, var,
                      os.path.join(args.imagenet_dir, "train"),
                      tp_dir, args, "train")
    if not args.skip_fp:
        process_split(vae, var,
                      os.path.join(args.imagenet_dir, "val"),
                      fp_dir, args, "val")

    print(f"\nDone.\n  TP -> {tp_dir}\n  FP -> {fp_dir}")


if __name__ == "__main__":
    main()
