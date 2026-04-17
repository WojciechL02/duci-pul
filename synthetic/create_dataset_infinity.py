#!/usr/bin/env python3
"""
Create synthetic ImageNet datasets using Infinity for dataset inference.

Produces:
  imagenet_tp_infinity_gtleak{G}_cfg{C}/
    train/  - IMAGENET TRAIN autoencoded through BSQ-VAE
    val/    - IMAGENET TRAIN img2img (first G scales real, rest generated)
    metadata.md

  imagenet_fp_infinity_gtleak{G}_cfg{C}/
    train/  - IMAGENET VAL autoencoded through BSQ-VAE
    val/    - IMAGENET VAL img2img
    metadata.md

With pn='0.06M' (256×256) Infinity has 7 scales.
gt_leak=3 keeps first 3 scales from real image, generates last 4.
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
sys.path.insert(0, os.path.join(SCRIPT_DIR, "Infinity"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "Infinity", "tools"))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_tensor

from run_infinity import (
    load_tokenizer, load_visual_tokenizer, load_transformer,
    encode_prompt, transform as infinity_transform,
)
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates

WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights", "Infinity")
CACHE_DIR = os.environ.get("INFINITY_CACHE_DIR", os.path.join(SCRIPT_DIR, ".cache"))


# ---------------------------------------------------------------------------
# WordNet prompt builder (Sariyildiz et al., "Fake it till you make it")
# ---------------------------------------------------------------------------
def build_wordnet_prompts(synset_ids):
    """Map ImageNet synset folder names to 'lemmas, definition' prompts.

    Uses the winning strategy from Sariyildiz et al. (arXiv:2212.08420):
    p_c = "c, d_c" where c = comma-separated lemma names and
    d_c = WordNet definition of the synset.
    """
    try:
        import nltk
        nltk.download("wordnet", quiet=True)
        from nltk.corpus import wordnet as wn
    except ImportError:
        print("[WARN] nltk not installed; falling back to generic prompt")
        return {sid: "a photo" for sid in synset_ids}

    prompts = {}
    for sid in synset_ids:
        try:
            offset = int(sid[1:])
            ss = wn.synset_from_pos_and_offset("n", offset)
            lemmas = ", ".join(l.replace("_", " ") for l in ss.lemma_names())
            defn = ss.definition()
            prompts[sid] = f"{lemmas}, {defn}"
        except Exception:
            prompts[sid] = sid
    return prompts


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ImageNetSubset(Dataset):
    def __init__(self, root, max_images=None):
        self.root = root
        self.samples = []
        synsets = sorted(
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d)) and d.startswith("n")
        )
        self.class_to_idx = {s: i for i, s in enumerate(synsets)}
        self.synset_ids = synsets
        for syn in synsets:
            syn_dir = os.path.join(root, syn)
            for fname in sorted(os.listdir(syn_dir)):
                if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    self.samples.append((
                        os.path.join(syn_dir, fname),
                        self.class_to_idx[syn], syn, fname))
            if max_images and len(self.samples) >= max_images:
                self.samples = self.samples[:max_images]
                break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Model loading helpers
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
        cache_dir=CACHE_DIR,
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
    """Replicate a B=1 text-condition tuple for B identical prompts."""
    kv_compact, lens, cu_seqlens_k, max_seqlen_k = text_cond_tuple
    kv_compact_B = kv_compact.repeat(B, 1)
    lens_B = lens * B
    L = lens[0]
    cu_B = torch.arange(
        B + 1, dtype=torch.int32, device=cu_seqlens_k.device) * L
    return (kv_compact_B, lens_B, cu_B, max_seqlen_k)


def stack_gt_bits(per_image_bits):
    """Stack list-of-per-image gt_ls_Bl into a single batched list."""
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
    """BSQ-VAE encode + decode.  Returns (recon_np [B,H,W,3], per_image_bits)."""
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
    """
    Batched Infinity img2img.  Calls autoregressive_infer_cfg with B>1.
    Returns uint8 numpy array [B, H, W, 3] RGB.
    """
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

    # img is [B, H, W, 3] uint8 BGR  →  RGB
    return img.flip(dims=(3,)).cpu().numpy()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def _save_one(img_np, path):
    Image.fromarray(img_np).save(path)


def write_metadata(out_dir, args, split_name, n_imgs, K, elapsed):
    with open(os.path.join(out_dir, "metadata.md"), "w") as f:
        f.write(f"# {os.path.basename(out_dir)}\n\n")
        f.write(f"**Source split**: `{split_name}` from "
                f"`{args.imagenet_dir}`\n\n")
        f.write("## Model\n\n")
        f.write("- Infinity-2B + BSQ-VAE (d32-reg) + Flan-T5-XL\n\n")
        f.write("## Hyperparameters\n\n| param | value |\n|---|---|\n")
        f.write(f"| total_scales | {K} |\n")
        f.write(f"| last_scales | {args.last_scales} (generate last "
                f"{args.last_scales} of {K}) |\n")
        f.write(f"| gt_leak | {args.gt_leak} (scales 0–"
                f"{args.gt_leak - 1} from real image) |\n")
        f.write(f"| cfg | {args.cfg} |\n| tau | {args.tau} |\n")
        f.write(f"| prompt | \"{args.prompt}\" |\n")
        f.write(f"| pn | {args.pn} |\n| seed | {args.seed} |\n")
        f.write(f"| batch_size | {args.batch_size} |\n\n")
        f.write("## Directories\n\n")
        f.write("- `train/` — BSQ-VAE autoencoder reconstruction\n")
        f.write(f"- `val/` — img2img (first {args.gt_leak} scales real, "
                f"last {args.last_scales} generated)\n\n")
        f.write(f"## Stats\n\n- images: {n_imgs}\n- time: {elapsed:.1f} s\n"
                f"- created: {datetime.datetime.now().isoformat()}\n")


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------
def process_split(infinity, vae, text_cond_dict, text_cond_fallback,
                  text_tokenizer, text_encoder,
                  split_dir, out_dir, scale_schedule, tgt_h, tgt_w,
                  inf_args, args, split_name):
    print(f"\n{'=' * 60}")
    print(f"Processing {split_name}: {split_dir} -> {out_dir}")
    print(f"{'=' * 60}")

    ds = ImageNetSubset(split_dir, max_images=args.max_images)
    K = len(scale_schedule)

    if not args.no_class_prompts and not text_cond_dict:
        wn_prompts = build_wordnet_prompts(ds.synset_ids)
        print(f"Encoding {len(wn_prompts)} class-specific prompts …")
        for sid, prompt_text in wn_prompts.items():
            text_cond_dict[sid] = encode_prompt(
                text_tokenizer, text_encoder, prompt_text)
        print(f"  example: {next(iter(wn_prompts.items()))}")

    train_dir = os.path.join(out_dir, "train")
    val_dir = os.path.join(out_dir, "val")
    executor = ThreadPoolExecutor(max_workers=8)
    all_futs = []

    total = len(ds)
    done = 0
    t0 = time.time()

    ae_bs = max(args.batch_size, 8)  # autoencoder can use larger batches
    gen_bs = args.batch_size

    for start in range(0, total, ae_bs):
        end = min(start + ae_bs, total)
        items = [ds[i] for i in range(start, end)]
        paths = [it[0] for it in items]
        syns = [it[2] for it in items]
        fnames = [it[3] for it in items]

        pil_images = [Image.open(p).convert("RGB") for p in paths]

        # --- Autoencoder pass (batched, fast) ---
        recon_np, per_image_bits = batch_autoencoder(
            vae, pil_images, scale_schedule, tgt_h, tgt_w, "cuda")

        for i in range(len(pil_images)):
            d = os.path.join(train_dir, syns[i])
            os.makedirs(d, exist_ok=True)
            stem = os.path.splitext(fnames[i])[0]
            if args.no_vqvae:
                raw_resized = pil_images[i].resize((tgt_w, tgt_h), Image.LANCZOS)
                all_futs.append(executor.submit(
                    _save_one, np.array(raw_resized), os.path.join(d, stem + ".png")))
            else:
                all_futs.append(executor.submit(
                    _save_one, recon_np[i], os.path.join(d, stem + ".png")))

        # --- img2img pass (batched) ---
        for j in range(0, len(pil_images), gen_bs):
            j_end = min(j + gen_bs, len(pil_images))
            sub_bits = per_image_bits[j:j_end]
            sub_syns = syns[j:j_end]
            sub_fnames = fnames[j:j_end]
            B = len(sub_bits)

            if not args.no_class_prompts and sub_syns[0] in text_cond_dict:
                batch_text_cond = text_cond_dict[sub_syns[0]]
            else:
                batch_text_cond = text_cond_fallback

            synth_np = batch_gen_img2img(
                infinity, vae, batch_text_cond, sub_bits,
                scale_schedule, inf_args,
                cfg=args.cfg, tau=args.tau,
                seed=args.seed + start + j,
            )

            for i in range(B):
                d = os.path.join(val_dir, sub_syns[i])
                os.makedirs(d, exist_ok=True)
                stem = os.path.splitext(sub_fnames[i])[0]
                all_futs.append(executor.submit(
                    _save_one, synth_np[i], os.path.join(d, stem + ".png")))

        done += len(pil_images)
        rate = done / (time.time() - t0)
        print(f"  [{done}/{total}] {rate:.2f} img/s")

    for fu in all_futs:
        fu.result()

    elapsed = time.time() - t0
    write_metadata(out_dir, args, split_name, total, K, elapsed)
    print(f"Finished {split_name}: {total} images in {elapsed:.1f} s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    pa = argparse.ArgumentParser(
        description="Create Infinity synthetic ImageNet datasets")
    pa.add_argument("--imagenet_dir", type=str,
                    default=os.path.join(SCRIPT_DIR, "..", "cdi", "data", "imagenet_5pc"))
    pa.add_argument("--output_root", type=str, default=os.path.join(SCRIPT_DIR, "datasets"))
    pa.add_argument("--last_scales", type=int, required=True,
                    help="Number of scales to generate (last N of K)")
    pa.add_argument("--cfg", type=float, default=3.0)
    pa.add_argument("--tau", type=float, default=0.5)
    pa.add_argument("--prompt", type=str, default="a photo")
    pa.add_argument("--pn", type=str, default="0.06M",
                    choices=["0.06M", "0.25M", "1M"])
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--batch_size", type=int, default=4,
                    help="Batch size for img2img generation")
    pa.add_argument("--max_images", type=int, default=None)
    pa.add_argument("--gpu", type=int, default=2)
    pa.add_argument("--skip_tp", action="store_true")
    pa.add_argument("--skip_fp", action="store_true")
    pa.add_argument("--no_vqvae", action="store_true",
                    help="Save raw resized/cropped images for train/ instead of BSQ-VAE reconstructions")
    pa.add_argument("--no_class_prompts", action="store_true",
                    help="Use a single generic prompt instead of per-class WordNet prompts")
    args = pa.parse_args()

    torch.cuda.set_device(0)

    scale_schedule = build_scale_schedule(args.pn)
    K = len(scale_schedule)
    assert 0 < args.last_scales <= K, f"last_scales must be 1–{K}"
    args.gt_leak = K - args.last_scales

    tgt_h = scale_schedule[-1][1] * 16
    tgt_w = scale_schedule[-1][2] * 16
    print(f"Scale schedule: {K} scales, target: {tgt_h}×{tgt_w}, "
          f"last_scales={args.last_scales}, gt_leak={args.gt_leak}")

    tag = f"infinity_last{args.last_scales}_cfg{args.cfg}"
    tp_dir = os.path.join(args.output_root, f"imagenet_tp_{tag}")
    fp_dir = os.path.join(args.output_root, f"imagenet_fp_{tag}")

    inf_args = get_infinity_args(args.pn)
    inf_args.gt_leak = args.gt_leak

    print("Loading Flan-T5-XL …")
    text_tokenizer, text_encoder = load_tokenizer(
        t5_path=inf_args.text_encoder_ckpt)
    print("Loading BSQ-VAE …")
    vae = load_visual_tokenizer(inf_args)
    print("Loading Infinity-2B …")
    infinity = load_transformer(vae, inf_args)
    print("Models ready.")

    # Encode fallback prompt (used when --class_prompts is off, or as default)
    text_cond_fallback = encode_prompt(text_tokenizer, text_encoder, args.prompt)
    text_cond_dict = {}  # populated lazily in process_split when --class_prompts

    if not args.no_class_prompts:
        print("Using per-class WordNet prompts (Sariyildiz et al. strategy)")
    else:
        print(f"Using single prompt: \"{args.prompt}\"")

    if not args.skip_tp:
        process_split(infinity, vae, text_cond_dict, text_cond_fallback,
                      text_tokenizer, text_encoder,
                      os.path.join(args.imagenet_dir, "train"),
                      tp_dir, scale_schedule, tgt_h, tgt_w,
                      inf_args, args, "train")
    if not args.skip_fp:
        process_split(infinity, vae, text_cond_dict, text_cond_fallback,
                      text_tokenizer, text_encoder,
                      os.path.join(args.imagenet_dir, "val"),
                      fp_dir, scale_schedule, tgt_h, tgt_w,
                      inf_args, args, "val")

    print(f"\nDone.\n  TP -> {tp_dir}\n  FP -> {fp_dir}")


if __name__ == "__main__":
    main()
