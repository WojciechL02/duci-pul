#!/usr/bin/env python3
"""
Create synthetic ImageNet datasets using Stable Diffusion 1.5 img2img.

Reconstruction of the deleted create_dataset_sd.py (see metadata.md of
imagenet_{tp,fp}_sd_strength*_cfg7.5 datasets), extended with --size to
support 512x512 targets (VAR-d36).

Produces:
  imagenet_tp_sd{SIZE}_strength{S}_cfg{G}/
    train/  - IMAGENET TRAIN autoencoded through SD VAE (sd-vae-ft-mse)
    val/    - IMAGENET TRAIN img2img (SD 1.5, class-name prompt)
    metadata.md

  imagenet_fp_sd{SIZE}_strength{S}_cfg{G}/
    train/  - IMAGENET VAL autoencoded through SD VAE
    val/    - IMAGENET VAL img2img
    metadata.md

For --size 256 the output dirs keep the legacy naming (imagenet_tp_sd_strength...).
"""
import argparse
import datetime
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# Must set CUDA_VISIBLE_DEVICES before importing torch
_gpu_arg = None
for i, a in enumerate(sys.argv):
    if a == "--gpu" and i + 1 < len(sys.argv):
        _gpu_arg = sys.argv[i + 1]
if _gpu_arg is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_arg

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "RAR"))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from imagenet_classes import imagenet_idx2classname
from img2img_sd import load_models, batch_ae_reconstruct, batch_img2img_sd


# ---------------------------------------------------------------------------
# Dataset (same conventions as create_dataset_var.py)
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
# I/O helpers
# ---------------------------------------------------------------------------
def _save_tensor(img_01, path):
    arr = (img_01.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _resize_batch(imgs_01, size, resample=Image.BICUBIC):
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    return torch.stack([
        to_tensor(to_pil(imgs_01[i].cpu()).resize((size, size), resample))
        for i in range(imgs_01.shape[0])
    ])


def _degrade_batch(imgs_01, r, size):
    """Bicubic round-trip size -> r -> size.

    Reproduces what SD received in the 256 pipeline: an image carrying r-worth
    of detail rendered on a `size` grid. Applied to the img2img branch only.
    """
    return _resize_batch(_resize_batch(imgs_01, r), size)


def _save_pil(img, path):
    img.save(path)


def write_metadata(out_dir, args, split_name, n_imgs, elapsed):
    with open(os.path.join(out_dir, "metadata.md"), "w") as f:
        f.write(f"# {os.path.basename(out_dir)}\n\n")
        f.write(f"**Source split**: `{split_name}` from `{args.imagenet_dir}`\n\n")
        f.write("## Model\n\n")
        f.write("- Stable Diffusion 1.5 (runwayml/stable-diffusion-v1-5) "
                "+ SD VAE (stabilityai/sd-vae-ft-mse)\n\n")
        f.write("## Hyperparameters\n\n")
        f.write("| param | value |\n|---|---|\n")
        f.write(f"| size | {args.size} |\n")
        f.write(f"| strength | {args.strength} |\n")
        f.write(f"| guidance_scale | {args.guidance_scale} |\n")
        f.write(f"| seed | {args.seed} |\n")
        f.write(f"| batch_size | {args.batch_size} |\n")
        f.write(f"| degrade_input | {args.degrade_input or 'none'} |\n")
        f.write(f"| emulate256 | {args.emulate256 or 'none'} |\n")
        f.write("| prompt | ImageNet class name |\n\n")
        f.write("## Directories\n\n")
        if args.skip_ae:
            f.write("- `train/` — SKIPPED (reuse the undegraded run)\n")
        else:
            f.write("- `train/` — SD VAE autoencoder reconstruction\n")
        f.write(f"- `val/` — img2img (Stable Diffusion, strength={args.strength})\n")
        if args.emulate256:
            R = args.emulate256
            f.write(f"\n256-pipeline emulation at R={R}, stored at {args.size}:\n"
                    f"- source: {args.size} -> {R} LANCZOS\n"
                    f"- AE: VAE@{R} -> {args.size} BICUBIC\n"
                    f"- img2img: {R} -> {args.size} -> SD@{args.size} -> {R} -> "
                    f"{args.size}, all BICUBIC\n")
        if args.degrade_input:
            f.write(f"  img2img input bicubic {args.size}->{args.degrade_input}->{args.size} "
                    "before SD; AE branch untouched\n")
        f.write("\n")
        f.write(f"## Stats\n\n- images: {n_imgs}\n- time: {elapsed:.1f} s\n"
                f"- created: {datetime.datetime.now().isoformat()}\n")


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------
def process_split(ae, pipe, split_dir, out_dir, args, split_name, device):
    print(f"\n{'=' * 60}")
    print(f"Processing {split_name}: {split_dir} -> {out_dir}")
    print(f"{'=' * 60}", flush=True)

    tf = transforms.Compose([
        transforms.Resize(args.size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(args.size),
        transforms.ToTensor(),
    ])

    ds = ImageNetSubset(split_dir, tf, max_images=args.max_images)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=min(4, args.batch_size), pin_memory=True,
        collate_fn=_collate,
    )

    train_dir = os.path.join(out_dir, "train")
    val_dir = os.path.join(out_dir, "val")
    executor = ThreadPoolExecutor(max_workers=8)
    all_futs = []

    total = len(ds)
    done = 0
    t0 = time.time()

    for bi, (imgs, labels, syns, fnames) in enumerate(loader):
        imgs = imgs.to(device)
        B = imgs.shape[0]
        stems = [os.path.splitext(f)[0] for f in fnames]
        for syn in set(syns):
            if not args.skip_ae:
                os.makedirs(os.path.join(train_dir, syn), exist_ok=True)
            os.makedirs(os.path.join(val_dir, syn), exist_ok=True)

        # In emulate256 mode both branches start from an R-resolution source, as
        # in the 256 pipeline; only the final store-at-`size` upsample is added.
        R = args.emulate256
        ae_input = _resize_batch(imgs, R, Image.LANCZOS).to(device) if R else imgs

        # train/ = SD VAE reconstruction
        if not args.skip_ae:
            recon = batch_ae_reconstruct(ae, ae_input, device)
            if R:
                recon = _resize_batch(recon, args.size)
            for i in range(B):
                all_futs.append(executor.submit(
                    _save_tensor, recon[i],
                    os.path.join(train_dir, syns[i], stems[i] + ".png"),
                ))

        # val/ = SD img2img with class-name prompt
        prompts = [imagenet_idx2classname[int(l)] for l in labels]
        if R:
            i2i_input = _resize_batch(ae_input, args.size).to(device)
        elif args.degrade_input:
            i2i_input = _degrade_batch(imgs, args.degrade_input, args.size).to(device)
        else:
            i2i_input = imgs
        synth = batch_img2img_sd(
            pipe, i2i_input, prompts,
            strength=args.strength, guidance_scale=args.guidance_scale,
            seed_base=args.seed + bi, device=device, size=args.size,
        )
        if R:
            synth = [im.resize((R, R), Image.BICUBIC)
                       .resize((args.size, args.size), Image.BICUBIC) for im in synth]
        for i in range(B):
            all_futs.append(executor.submit(
                _save_pil, synth[i],
                os.path.join(val_dir, syns[i], stems[i] + ".png"),
            ))

        done += B
        rate = done / (time.time() - t0)
        print(f"  [{done}/{total}] {rate:.2f} img/s", flush=True)

    for fu in all_futs:
        fu.result()

    elapsed = time.time() - t0
    write_metadata(out_dir, args, split_name, total, elapsed)
    print(f"Finished {split_name}: {total} images in {elapsed:.1f} s "
          f"({total / max(elapsed, 1e-6):.2f} img/s)", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    pa = argparse.ArgumentParser(
        description="Create SD img2img synthetic ImageNet datasets")
    pa.add_argument("--imagenet_dir", type=str,
                    default="/data/full-pipeline/data/imagenet_5pc")
    pa.add_argument("--output_root", type=str,
                    default=os.path.join(SCRIPT_DIR, "datasets"))
    pa.add_argument("--size", type=int, default=256,
                    help="Output image resolution (512 for VAR-d36 targets)")
    pa.add_argument("--strength", type=float, default=0.5)
    pa.add_argument("--guidance_scale", type=float, default=7.5)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--batch_size", type=int, default=16)
    pa.add_argument("--max_images", type=int, default=None)
    pa.add_argument("--gpu", type=int, default=0)
    pa.add_argument("--skip_tp", action="store_true")
    pa.add_argument("--skip_fp", action="store_true")
    pa.add_argument("--skip_ae", action="store_true",
                    help="Skip the train/ AE branch (reuse it from an undegraded run)")
    pa.add_argument("--degrade_input", type=int, default=None,
                    help="Bicubic round-trip the img2img input through this resolution")
    pa.add_argument("--emulate256", type=int, default=None,
                    help="Run the whole 256-pipeline geometry at this resolution, "
                         "storing the result upsampled to --size")
    pa.add_argument("--test_2images", action="store_true",
                    help="Smoke test: process only 2 images per split")
    pa.add_argument("--tag_suffix", type=str, default="",
                    help="Appended to size tag, e.g. 'var' -> sd512var_...")
    args = pa.parse_args()

    if args.test_2images:
        args.max_images = 2
        args.batch_size = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    size_tag = "sd" if args.size == 256 else f"sd{args.size}"
    if args.tag_suffix:
        size_tag = f"{size_tag}{args.tag_suffix}"
    tag = f"{size_tag}_strength{args.strength}_cfg{args.guidance_scale}"
    tp_dir = os.path.join(args.output_root, f"imagenet_tp_{tag}")
    fp_dir = os.path.join(args.output_root, f"imagenet_fp_{tag}")

    print("Loading SD VAE and img2img pipeline…", flush=True)
    ae, pipe = load_models(device)
    pipe.set_progress_bar_config(disable=True)
    print("Models ready.", flush=True)

    if not args.skip_tp:
        process_split(ae, pipe,
                      os.path.join(args.imagenet_dir, "train"),
                      tp_dir, args, "train", device)
    if not args.skip_fp:
        process_split(ae, pipe,
                      os.path.join(args.imagenet_dir, "val"),
                      fp_dir, args, "val", device)

    print(f"\nDone.\n  TP -> {tp_dir}\n  FP -> {fp_dir}")


if __name__ == "__main__":
    main()
