#!/usr/bin/env python3
"""
Create synthetic COCO datasets using Infinity for dataset inference.

Produces self-contained datasets with images, annotations, and text embeddings:

  coco_tp_infinity_last{N}_cfg{C}/
    train2014/      - COCO TRAIN autoencoded through BSQ-VAE (or raw if --no_vqvae)
    val2014/        - COCO TRAIN img2img (first G scales real, rest generated)
    annotations/    - captions_train2014.json and captions_val2014.json (both from train)
    train_text_emb/ - CLIP embeddings (from train)
    val_text_emb/   - CLIP embeddings (from train, copied)
    metadata.md

  coco_fp_infinity_last{N}_cfg{C}/
    train2014/      - COCO VAL autoencoded (or raw)
    val2014/        - COCO VAL img2img
    annotations/    - captions_train2014.json and captions_val2014.json (both from val)
    train_text_emb/ - CLIP embeddings (from val)
    val_text_emb/   - CLIP embeddings (from val, copied)
    metadata.md

Annotations and embeddings always correspond to the SOURCE split, not the output
subdir name, so the CDI UViT dataloader sees consistent data in both subdirs.
"""
import argparse
import json
import os
import shutil
import sys
import time
import datetime
from concurrent.futures import ThreadPoolExecutor

_gpu_arg = None
for i, a in enumerate(sys.argv):
    if a == "--gpu" and i + 1 < len(sys.argv):
        _gpu_arg = sys.argv[i + 1]
if _gpu_arg is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_arg

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "Infinity"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "Infinity", "tools"))

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from run_infinity import (
    load_tokenizer, load_visual_tokenizer, load_transformer,
    encode_prompt, transform as infinity_transform,
)
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates

WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights", "Infinity")


# ---------------------------------------------------------------------------
# COCO Dataset
# ---------------------------------------------------------------------------
class COCOSubset(Dataset):
    """Flat COCO image loader with caption support.

    Items: (path, index, filename, captions)
    where index is the position in sorted(coco.imgs.keys()), matching
    the text embedding file naming convention ({index}_{caption_idx}.npy).
    """

    def __init__(self, img_dir, ann_file, max_images=None):
        from pycocotools.coco import COCO
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.keys = list(sorted(self.coco.imgs.keys()))

        if max_images and max_images < len(self.keys):
            self.keys = self.keys[:max_images]

        self.samples = []
        for idx, key in enumerate(self.keys):
            info = self.coco.loadImgs(key)[0]
            fname = info["file_name"]
            anns = self.coco.loadAnns(self.coco.getAnnIds(key))
            captions = [a["caption"] for a in anns]
            self.samples.append((
                os.path.join(img_dir, fname),
                idx,
                fname,
                captions,
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Model loading (shared with ImageNet script)
# ---------------------------------------------------------------------------
def get_infinity_args(pn="0.06M"):
    return argparse.Namespace(
        pn=pn,
        model_path=os.path.join(WEIGHTS_DIR, "infinity_2b_reg.pth"),
        cfg_insertion_layer=0,
        vae_type=32,
        vae_path=os.path.join(WEIGHTS_DIR, "infinity_vae_d32reg.pth"),
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        model_type="infinity_2b",
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0,
        sampling_per_bits=1,
        text_encoder_ckpt=os.path.join(WEIGHTS_DIR, "flan-t5-xl"),
        text_channels=2048,
        apply_spatial_patchify=0,
        h_div_w_template=1.000,
        use_flex_attn=0,
        cache_dir="/dev/shm",
        checkpoint_type="torch",
        seed=42,
        bf16=0,
        save_file="tmp.jpg",
        enable_model_cache=0,
    )


def build_scale_schedule(pn="0.06M"):
    h_div_w = 1.0
    t = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
    schedule = dynamic_resolution_h_w[t][pn]["scales"]
    return [(1, h, w) for (_, h, w) in schedule]


# ---------------------------------------------------------------------------
# Batched text conditioning helpers
# ---------------------------------------------------------------------------
def replicate_text_cond(text_cond_tuple, B):
    kv_compact, lens, cu_seqlens_k, max_seqlen_k = text_cond_tuple
    kv_compact_B = kv_compact.repeat(B, 1)
    lens_B = lens * B
    L = lens[0]
    cu_B = torch.arange(
        B + 1, dtype=torch.int32, device=cu_seqlens_k.device) * L
    return (kv_compact_B, lens_B, cu_B, max_seqlen_k)


def stack_gt_bits(per_image_bits):
    B = len(per_image_bits)
    n_scales = len(per_image_bits[0])
    return [
        torch.cat([per_image_bits[i][si] for i in range(B)], dim=0)
        for si in range(n_scales)
    ]


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------
@torch.no_grad()
def batch_autoencoder(vae, pil_images, scale_schedule, tgt_h, tgt_w, device):
    tensors = [infinity_transform(img, tgt_h, tgt_w) for img in pil_images]
    batch = torch.stack(tensors).to(device)
    schedule = [(s[0], s[1], s[2]) for s in scale_schedule]
    h, z, _, all_bit_indices, _, _ = vae.encode(batch, scale_schedule=schedule)
    recon = vae.decode(z)
    B = batch.shape[0]
    per_image_bits = [
        [ab[i:i + 1] for ab in all_bit_indices] for i in range(B)
    ]
    recon_np = ((recon + 1) / 2).permute(0, 2, 3, 1).clamp(0, 1)
    return (recon_np.cpu().numpy() * 255).astype(np.uint8), per_image_bits


@torch.no_grad()
def batch_gen_img2img(infinity, vae, text_cond_b1, per_image_bits,
                      scale_schedule, inf_args, cfg, tau, seed):
    B = len(per_image_bits)
    text_cond_B = replicate_text_cond(text_cond_b1, B)
    gt_ls_Bl = stack_gt_bits(per_image_bits)
    cfg_list = [cfg] * len(scale_schedule)
    tau_list = [tau] * len(scale_schedule)
    with torch.cuda.amp.autocast(
            enabled=True, dtype=torch.bfloat16, cache_enabled=True):
        _, _, img = infinity.autoregressive_infer_cfg(
            vae=vae,
            scale_schedule=scale_schedule,
            label_B_or_BLT=text_cond_B,
            g_seed=seed,
            B=B,
            negative_label_B_or_BLT=None,
            force_gt_Bhw=None,
            cfg_sc=3,
            cfg_list=cfg_list,
            tau_list=tau_list,
            top_k=900,
            top_p=0.97,
            returns_vemb=1,
            ratio_Bl1=None,
            gumbel=0,
            norm_cfg=False,
            cfg_exp_k=0.0,
            cfg_insertion_layer=[inf_args.cfg_insertion_layer],
            vae_type=inf_args.vae_type,
            softmax_merge_topk=-1,
            ret_img=True,
            trunk_scale=1000,
            gt_leak=inf_args.gt_leak,
            gt_ls_Bl=gt_ls_Bl,
            inference_mode=True,
            sampling_per_bits=inf_args.sampling_per_bits,
        )
    return img.flip(dims=(3,)).cpu().numpy()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def _save_one(img_np, path):
    Image.fromarray(img_np).save(path)


def bundle_annotations_and_embeddings(out_dir, coco_dir, source_split):
    """Copy source split's annotations and embeddings into both train/val slots."""
    src_ann = os.path.join(coco_dir, "annotations",
                           f"captions_{source_split}2014.json")
    src_emb = os.path.join(coco_dir, f"{source_split}_text_emb")

    ann_dir = os.path.join(out_dir, "annotations")
    os.makedirs(ann_dir, exist_ok=True)

    for target_name in ["captions_train2014.json", "captions_val2014.json"]:
        dst = os.path.join(ann_dir, target_name)
        if not os.path.exists(dst):
            shutil.copy2(src_ann, dst)
            print(f"  Copied annotations -> {target_name}")

    for target_dir in ["train_text_emb", "val_text_emb"]:
        dst = os.path.join(out_dir, target_dir)
        if not os.path.exists(dst):
            shutil.copytree(src_emb, dst)
            print(f"  Copied embeddings -> {target_dir}/")


def write_metadata(out_dir, args, split_name, n_imgs, K, elapsed):
    with open(os.path.join(out_dir, "metadata.md"), "w") as f:
        f.write(f"# {os.path.basename(out_dir)}\n\n")
        f.write(f"**Source**: COCO `{split_name}` from `{args.coco_dir}`\n\n")
        f.write("## Model\n\n")
        f.write("- Infinity-2B + BSQ-VAE (d32-reg) + Flan-T5-XL\n\n")
        f.write("## Hyperparameters\n\n| param | value |\n|---|---|\n")
        f.write(f"| total_scales | {K} |\n")
        f.write(f"| last_scales | {args.last_scales} (generate last "
                f"{args.last_scales} of {K}) |\n")
        f.write(f"| gt_leak | {args.gt_leak} (scales 0-"
                f"{args.gt_leak - 1} from real image) |\n")
        f.write(f"| cfg | {args.cfg} |\n| tau | {args.tau} |\n")
        f.write(f"| prompts | per-image captions from COCO |\n")
        f.write(f"| pn | {args.pn} |\n| seed | {args.seed} |\n")
        f.write(f"| batch_size | {args.batch_size} |\n\n")
        f.write("## Directories\n\n")
        f.write("- `train2014/` - BSQ-VAE autoencoder reconstruction "
                "(or raw if --no_vqvae)\n")
        f.write(f"- `val2014/` - img2img (first {args.gt_leak} scales real, "
                f"last {args.last_scales} generated)\n")
        f.write("- `annotations/` - COCO captions JSON (from source split)\n")
        f.write("- `train_text_emb/`, `val_text_emb/` - "
                "CLIP embeddings (from source split)\n\n")
        f.write(f"## Stats\n\n- images: {n_imgs}\n- time: {elapsed:.1f} s\n"
                f"- created: {datetime.datetime.now().isoformat()}\n")


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------
def process_split(infinity, vae, text_tokenizer, text_encoder,
                  text_cond_fallback, coco_dir, source_split,
                  out_dir, scale_schedule, tgt_h, tgt_w,
                  inf_args, args):
    img_dir = os.path.join(coco_dir, f"{source_split}2014")
    ann_file = os.path.join(coco_dir, "annotations",
                            f"captions_{source_split}2014.json")

    print(f"\n{'=' * 60}")
    print(f"Processing COCO {source_split}: {img_dir} -> {out_dir}")
    print(f"{'=' * 60}")

    ds = COCOSubset(img_dir, ann_file, max_images=args.max_images)
    K = len(scale_schedule)

    # Pre-encode per-image caption prompts
    text_cond_cache = {}
    if not args.no_caption_prompts:
        print(f"Encoding per-image caption prompts for {len(ds)} images ...")
        for i in range(len(ds)):
            _, idx, fname, captions = ds[i]
            prompt = captions[0] if captions else "a photo"
            text_cond_cache[idx] = encode_prompt(
                text_tokenizer, text_encoder, prompt)
            if i == 0:
                print(f"  example [{idx}]: \"{prompt[:80]}\"")
        print(f"  Encoded {len(text_cond_cache)} prompts")

    train_dir = os.path.join(out_dir, "train2014")
    val_dir = os.path.join(out_dir, "val2014")
    executor = ThreadPoolExecutor(max_workers=8)
    all_futs = []

    total = len(ds)
    done = 0
    t0 = time.time()

    ae_bs = max(args.batch_size, 8)
    gen_bs = args.batch_size

    for start in range(0, total, ae_bs):
        end = min(start + ae_bs, total)
        items = [ds[i] for i in range(start, end)]
        paths = [it[0] for it in items]
        indices = [it[1] for it in items]
        fnames = [it[2] for it in items]

        pil_images = [Image.open(p).convert("RGB") for p in paths]

        # --- Autoencoder pass ---
        recon_np, per_image_bits = batch_autoencoder(
            vae, pil_images, scale_schedule, tgt_h, tgt_w, "cuda")

        for i in range(len(pil_images)):
            os.makedirs(train_dir, exist_ok=True)
            stem = os.path.splitext(fnames[i])[0]
            out_path = os.path.join(train_dir, stem + ".png")
            if args.no_vqvae:
                raw = pil_images[i].resize((tgt_w, tgt_h), Image.LANCZOS)
                all_futs.append(executor.submit(
                    _save_one, np.array(raw), out_path))
            else:
                all_futs.append(executor.submit(
                    _save_one, recon_np[i], out_path))

        # --- img2img pass ---
        for j in range(0, len(pil_images), gen_bs):
            j_end = min(j + gen_bs, len(pil_images))
            sub_bits = per_image_bits[j:j_end]
            sub_indices = indices[j:j_end]
            sub_fnames = fnames[j:j_end]
            B = len(sub_bits)

            if not args.no_caption_prompts and sub_indices[0] in text_cond_cache:
                batch_text_cond = text_cond_cache[sub_indices[0]]
            else:
                batch_text_cond = text_cond_fallback

            synth_np = batch_gen_img2img(
                infinity, vae, batch_text_cond, sub_bits,
                scale_schedule, inf_args,
                cfg=args.cfg, tau=args.tau,
                seed=args.seed + start + j,
            )

            for i in range(B):
                os.makedirs(val_dir, exist_ok=True)
                stem = os.path.splitext(sub_fnames[i])[0]
                all_futs.append(executor.submit(
                    _save_one, synth_np[i],
                    os.path.join(val_dir, stem + ".png")))

        done += len(pil_images)
        rate = done / (time.time() - t0)
        print(f"  [{done}/{total}] {rate:.2f} img/s")

    for fu in all_futs:
        fu.result()

    # Bundle annotations and embeddings from the source split
    print("Bundling annotations and embeddings ...")
    bundle_annotations_and_embeddings(out_dir, coco_dir, source_split)

    elapsed = time.time() - t0
    write_metadata(out_dir, args, source_split, total, K, elapsed)
    print(f"Finished COCO {source_split}: {total} images in {elapsed:.1f} s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    pa = argparse.ArgumentParser(
        description="Create Infinity synthetic COCO datasets")
    pa.add_argument("--coco_dir", type=str,
                    default=os.path.join(SCRIPT_DIR, "..", "cdi", "data", "coco_5000"))
    pa.add_argument("--output_root", type=str,
                    default=os.path.join(SCRIPT_DIR, "datasets_coco"))
    pa.add_argument("--last_scales", type=int, required=True,
                    help="Number of scales to generate (last N of K)")
    pa.add_argument("--cfg", type=float, default=3.0)
    pa.add_argument("--tau", type=float, default=0.5)
    pa.add_argument("--prompt", type=str, default="a photo",
                    help="Fallback prompt when --no_caption_prompts is set")
    pa.add_argument("--pn", type=str, default="0.06M",
                    choices=["0.06M", "0.25M", "1M"])
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--batch_size", type=int, default=4)
    pa.add_argument("--max_images", type=int, default=None)
    pa.add_argument("--gpu", type=int, default=0)
    pa.add_argument("--skip_tp", action="store_true")
    pa.add_argument("--skip_fp", action="store_true")
    pa.add_argument("--no_vqvae", action="store_true",
                    help="Save raw resized images for train2014/ instead of "
                         "BSQ-VAE reconstructions")
    pa.add_argument("--no_caption_prompts", action="store_true",
                    help="Use a single generic prompt instead of per-image "
                         "COCO captions")
    args = pa.parse_args()

    torch.cuda.set_device(0)

    scale_schedule = build_scale_schedule(args.pn)
    K = len(scale_schedule)
    assert 0 < args.last_scales <= K, f"last_scales must be 1-{K}"
    args.gt_leak = K - args.last_scales

    tgt_h = scale_schedule[-1][1] * 16
    tgt_w = scale_schedule[-1][2] * 16
    print(f"Scale schedule: {K} scales, target: {tgt_h}x{tgt_w}, "
          f"last_scales={args.last_scales}, gt_leak={args.gt_leak}")

    tag = f"infinity_last{args.last_scales}_cfg{args.cfg}"
    tp_dir = os.path.join(args.output_root, f"coco_tp_{tag}")
    fp_dir = os.path.join(args.output_root, f"coco_fp_{tag}")

    inf_args = get_infinity_args(args.pn)
    inf_args.gt_leak = args.gt_leak

    print("Loading Flan-T5-XL ...")
    text_tokenizer, text_encoder = load_tokenizer(
        t5_path=inf_args.text_encoder_ckpt)
    print("Loading BSQ-VAE ...")
    vae = load_visual_tokenizer(inf_args)
    print("Loading Infinity-2B ...")
    infinity = load_transformer(vae, inf_args)
    print("Models ready.")

    text_cond_fallback = encode_prompt(text_tokenizer, text_encoder, args.prompt)

    if not args.no_caption_prompts:
        print("Using per-image COCO caption prompts")
    else:
        print(f"Using single prompt: \"{args.prompt}\"")

    if not args.skip_tp:
        process_split(infinity, vae, text_tokenizer, text_encoder,
                      text_cond_fallback, args.coco_dir, "train",
                      tp_dir, scale_schedule, tgt_h, tgt_w,
                      inf_args, args)
    if not args.skip_fp:
        process_split(infinity, vae, text_tokenizer, text_encoder,
                      text_cond_fallback, args.coco_dir, "val",
                      fp_dir, scale_schedule, tgt_h, tgt_w,
                      inf_args, args)

    print(f"\nDone.\n  TP -> {tp_dir}\n  FP -> {fp_dir}")


if __name__ == "__main__":
    main()
