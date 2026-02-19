#!/usr/bin/env python3
"""
Create synthetic ImageNet datasets using RAR for dataset inference.

Produces:
  imagenet_tp_rar_gen{R}_{MODE}_cfg{C}/
    train/  - IMAGENET TRAIN autoencoded through MaskGIT-VQGAN
    val/    - IMAGENET TRAIN img2img
    metadata.md

  imagenet_fp_rar_gen{R}_{MODE}_cfg{C}/
    train/  - IMAGENET VAL autoencoded through MaskGIT-VQGAN
    val/    - IMAGENET VAL img2img
    metadata.md

RAR tokenises images to 256 discrete tokens (16×16 raster grid).

Modes:
  suffix (default) — first (1-gen_ratio)*256 tokens are real, last gen_ratio*256
      tokens are model-generated.  The model conditions on the real prefix via
      KV cache, so the transition is seamless.
  random — each token independently has gen_ratio probability of being
      model-generated (original approach, may cause micro-inconsistencies).
"""
import argparse
import os
import sys
import time
import datetime
from concurrent.futures import ThreadPoolExecutor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "RAR"))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import demo_util
from utils.train_utils import create_pretrained_tokenizer

WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights", "RAR")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ImageNetSubset(Dataset):
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
    rar_dir = os.path.join(SCRIPT_DIR, "RAR")
    config = demo_util.get_config(
        os.path.join(rar_dir, "configs", "training", "generator", "rar.yaml")
    )
    config.experiment.generator_checkpoint = os.path.join(
        WEIGHTS_DIR, "rar_xxl.bin")
    config.model.vq_model.pretrained_tokenizer_weight = os.path.join(
        WEIGHTS_DIR, "maskgit-vqgan-imagenet-f16-256.bin")
    config.model.generator.hidden_size = 1408
    config.model.generator.num_hidden_layers = 40
    config.model.generator.num_attention_heads = 16
    config.model.generator.intermediate_size = 6144

    tokenizer = create_pretrained_tokenizer(config)
    tokenizer.to(device)
    generator = demo_util.get_rar_generator(config)
    generator.to(device)
    return tokenizer, generator


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------
@torch.no_grad()
def batch_autoencoder(tokenizer, imgs):
    """MaskGIT-VQGAN reconstruction.  imgs: [B,3,256,256] in [0,1] -> [0,1]."""
    codes = tokenizer.encode(imgs)
    if codes.dim() == 3:
        codes = codes.view(codes.shape[0], -1)
    return torch.clamp(tokenizer.decode(codes), 0, 1)


@torch.no_grad()
def batch_img2img(generator, tokenizer, imgs, labels,
                  gen_ratio, mode, guidance_scale, temperature,
                  guidance_pow, seed):
    """
    Batched RAR img2img.

    suffix mode: first (1-gen_ratio)*256 tokens = real,
                 last gen_ratio*256 tokens = model-generated.
    random mode: each token independently has gen_ratio probability
                 of being model-generated.

    imgs: [B,3,256,256] in [0,1], labels: [B] -> [0,1].
    """
    device = imgs.device
    torch.manual_seed(seed)

    real_tokens = tokenizer.encode(imgs)
    if real_tokens.dim() == 3:
        real_tokens = real_tokens.view(real_tokens.shape[0], -1)

    seq_len = generator.image_seq_len
    n_real = int((1 - gen_ratio) * seq_len) if mode == "suffix" else 0

    condition = generator.preprocess_condition(labels, cond_drop_prob=0.0)
    num_samples = condition.shape[0]
    ids = torch.full((num_samples, 0), -1, device=device, dtype=torch.long)

    generator.enable_kv_cache()

    for step in range(seq_len):
        scale_pow = torch.ones((1,), device=device) * guidance_pow
        scale_step = (
            1 - torch.cos(
                ((step / seq_len) ** scale_pow) * torch.pi
            )
        ) * 0.5
        cfg_scale = (guidance_scale - 1) * scale_step + 1

        if guidance_scale != 0:
            logits = generator.forward_fn(
                torch.cat([ids, ids], dim=0),
                torch.cat([condition,
                           generator.get_none_condition(condition)], dim=0),
                orders=None, is_sampling=True,
            )
            cond_l, uncond_l = logits[:num_samples], logits[num_samples:]
            logits = uncond_l + (cond_l - uncond_l) * cfg_scale
        else:
            logits = generator.forward_fn(
                ids, condition, orders=None, is_sampling=True)

        logits = logits[:, -1] / temperature
        probs = F.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)

        if mode == "suffix":
            # Real prefix → generated suffix
            token = real_tokens[:, step:step + 1] if step < n_real else sampled
        else:
            # Random per-token replacement
            use_gen = torch.rand(num_samples, 1, device=device) < gen_ratio
            token = torch.where(use_gen, sampled, real_tokens[:, step:step + 1])

        ids = torch.cat((ids, token), dim=-1)

    generator.disable_kv_cache()
    return torch.clamp(tokenizer.decode(ids), 0, 1)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def _save_one(img_np, path):
    Image.fromarray(img_np).save(path)


def save_batch(tensor_01, syns, fnames, out_root, executor):
    arr = (tensor_01.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    futs = []
    for i in range(arr.shape[0]):
        d = os.path.join(out_root, syns[i])
        os.makedirs(d, exist_ok=True)
        stem = os.path.splitext(fnames[i])[0]
        futs.append(executor.submit(
            _save_one, arr[i], os.path.join(d, stem + ".png")))
    return futs


def write_metadata(out_dir, args, split_name, n_imgs, elapsed):
    n_gen = int(args.gen_ratio * 256)
    with open(os.path.join(out_dir, "metadata.md"), "w") as f:
        f.write(f"# {os.path.basename(out_dir)}\n\n")
        f.write(f"**Source split**: `{split_name}` from "
                f"`{args.imagenet_dir}`\n\n")
        f.write("## Model\n\n")
        f.write("- RAR-XXL (1.4 B params) + MaskGIT-VQGAN tokeniser\n\n")
        f.write("## Hyperparameters\n\n")
        f.write("| param | value |\n|---|---|\n")
        f.write(f"| gen_ratio | {args.gen_ratio} |\n")
        f.write(f"| mode | {args.mode} |\n")
        if args.mode == "suffix":
            f.write(f"| real_tokens | {256 - n_gen} / 256 (positions 0–"
                    f"{255 - n_gen}) |\n")
            f.write(f"| generated_tokens | {n_gen} / 256 (positions "
                    f"{256 - n_gen}–255) |\n")
        else:
            f.write(f"| per-token gen prob | {args.gen_ratio} |\n")
        f.write(f"| guidance_scale | {args.guidance_scale} |\n")
        f.write(f"| temperature | {args.temperature} |\n")
        f.write(f"| guidance_pow | {args.guidance_pow} |\n")
        f.write(f"| seed | {args.seed} |\n")
        f.write(f"| batch_size | {args.batch_size} |\n\n")
        f.write("## Directories\n\n")
        f.write("- `train/` — MaskGIT-VQGAN autoencoder reconstruction\n")
        f.write(f"- `val/` — img2img ({args.mode} mode, "
                f"{args.gen_ratio*100:.0f} % tokens generated)\n\n")
        f.write("## Note on teacher forcing\n\n")
        if args.mode == "suffix":
            f.write("Suffix mode: the first 90 % of the 16×16 raster-order "
                    "tokens are injected from the real image. The model "
                    "conditions on this real prefix through its KV cache and "
                    "generates the remaining tokens autoregressively. This "
                    "avoids the micro-inconsistencies that random token "
                    "replacement can introduce.\n\n")
        else:
            f.write("Random mode: at each of the 256 AR steps the model "
                    "predicts a token; with probability gen_ratio that "
                    "prediction is used, otherwise the real image's token "
                    "is injected.  The model always conditions on the full "
                    "sequence of previously placed tokens (real or "
                    "generated).\n\n")
        f.write(f"## Stats\n\n- images: {n_imgs}\n- time: {elapsed:.1f} s\n"
                f"- created: {datetime.datetime.now().isoformat()}\n")


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------
def process_split(tokenizer, generator, split_dir, out_dir, args, split_name):
    print(f"\n{'=' * 60}")
    print(f"Processing {split_name}: {split_dir} -> {out_dir}")
    print(f"{'=' * 60}")

    tf = transforms.Compose([
        transforms.Resize(256,
                          interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])

    ds = ImageNetSubset(split_dir, tf, max_images=args.max_images)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=min(4, args.batch_size), pin_memory=True,
        collate_fn=_collate,
    )

    device = next(generator.parameters()).device
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

        recon = batch_autoencoder(tokenizer, imgs)
        all_futs += save_batch(recon, syns, fnames, train_dir, executor)

        synth = batch_img2img(
            generator, tokenizer, imgs, labels,
            gen_ratio=args.gen_ratio,
            mode=args.mode,
            guidance_scale=args.guidance_scale,
            temperature=args.temperature,
            guidance_pow=args.guidance_pow,
            seed=args.seed + bi,
        )
        all_futs += save_batch(synth, syns, fnames, val_dir, executor)

        done += B
        rate = done / (time.time() - t0)
        print(f"  [{done}/{total}] {rate:.2f} img/s")

    for fu in all_futs:
        fu.result()

    elapsed = time.time() - t0
    write_metadata(out_dir, args, split_name, total, elapsed)
    print(f"Finished {split_name}: {total} images in {elapsed:.1f} s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    pa = argparse.ArgumentParser(
        description="Create RAR synthetic ImageNet datasets")
    pa.add_argument("--imagenet_dir", type=str, required=True)
    pa.add_argument("--output_root", type=str, default=".")
    pa.add_argument("--gen_ratio", type=float, default=0.1,
                    help="Fraction of tokens that are model-generated "
                         "(0.1 = last 10%% generated in suffix mode)")
    pa.add_argument("--mode", type=str, default="suffix",
                    choices=["suffix", "random"],
                    help="suffix: generate last N%% of tokens; "
                         "random: per-token replacement")
    pa.add_argument("--guidance_scale", type=float, default=8.0)
    pa.add_argument("--temperature", type=float, default=1.02)
    pa.add_argument("--guidance_pow", type=float, default=1.2)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--batch_size", type=int, default=8)
    pa.add_argument("--max_images", type=int, default=None)
    pa.add_argument("--gpu", type=int, default=2)
    pa.add_argument("--skip_tp", action="store_true")
    pa.add_argument("--skip_fp", action="store_true")
    args = pa.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda"

    tag = f"rar_gen{args.gen_ratio}_{args.mode}_cfg{args.guidance_scale}"
    tp_dir = os.path.join(args.output_root, f"imagenet_tp_{tag}")
    fp_dir = os.path.join(args.output_root, f"imagenet_fp_{tag}")

    print("Loading RAR-XXL + MaskGIT-VQGAN …")
    tokenizer, generator = load_models(device)
    print("Models ready.")

    if not args.skip_tp:
        process_split(tokenizer, generator,
                      os.path.join(args.imagenet_dir, "train"),
                      tp_dir, args, "train")
    if not args.skip_fp:
        process_split(tokenizer, generator,
                      os.path.join(args.imagenet_dir, "val"),
                      fp_dir, args, "val")

    print(f"\nDone.\n  TP -> {tp_dir}\n  FP -> {fp_dir}")


if __name__ == "__main__":
    main()
