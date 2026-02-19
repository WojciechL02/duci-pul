"""
img2img for VAR (Visual Autoregressive Modeling).

Encodes a real image through VAR's multi-scale VQVAE, keeps the first
n_real_scales token maps from the real image, and regenerates the
remaining scales with the VAR transformer.

VAR uses 10 scales: patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16).
n_real_scales=0 -> fully synthetic, n_real_scales=10 -> VQVAE reconstruction.
"""
import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "VAR"))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

import dist
from models import build_vae_var
from models.helpers import sample_with_top_k_top_p_

WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights", "VAR")
VAE_CKPT = os.path.join(WEIGHTS_DIR, "vae_ch160v4096z32.pth")
VAR_CKPT = os.path.join(WEIGHTS_DIR, "var_d30.pth")


def preprocess_image(image_path, size=256):
    img = Image.open(image_path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return tf(img).unsqueeze(0)


def tensor_to_pil(t):
    t = t.clamp(0, 1)
    return Image.fromarray((t[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))


@torch.no_grad()
def img2img_var(var_model, vae, img_tensor, class_label, n_real_scales=5,
                cfg=1.5, top_k=900, top_p=0.96, seed=42):
    """
    Generate an img2img variant using VAR.

    For the first n_real_scales, inject real image tokens.
    For the remaining scales, sample from the VAR transformer.
    """
    device = img_tensor.device
    B = 1
    patch_nums = var_model.patch_nums

    var_model.rng.manual_seed(seed)
    rng = var_model.rng

    # Encode real image -> per-scale token indices
    real_idx_Bls = vae.img_to_idxBl(img_tensor)

    # Setup class conditioning (same as autoregressive_infer_cfg)
    label_B = torch.full((B,), fill_value=class_label, device=device)
    sos = cond_BD = var_model.class_emb(
        torch.cat((label_B, torch.full_like(label_B, fill_value=var_model.num_classes)), dim=0)
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

        cond_BD_or_gss = var_model.shared_ada_lin(cond_BD)
        x = next_token_map
        for b in var_model.blocks:
            x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
        logits_BlV = var_model.get_logits(x, cond_BD)

        t = cfg * ratio
        logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]

        if si < n_real_scales:
            # Use real image tokens for this scale
            idx_Bl = real_idx_Bls[si].to(device)
        else:
            # Sample from model
            idx_Bl = sample_with_top_k_top_p_(
                logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1
            )[:, :, 0]

        h_BChw = var_model.vae_quant_proxy[0].embedding(idx_Bl)
        h_BChw = h_BChw.transpose_(1, 2).reshape(B, var_model.Cvae, pn, pn)
        f_hat, next_token_map = var_model.vae_quant_proxy[0].get_next_autoregressive_input(
            si, len(patch_nums), f_hat, h_BChw
        )
        if si != var_model.num_stages_minus_1:
            next_token_map = next_token_map.view(B, var_model.Cvae, -1).transpose(1, 2)
            next_token_map = (
                var_model.word_embed(next_token_map)
                + lvl_pos[:, cur_L : cur_L + patch_nums[si + 1] ** 2]
            )
            next_token_map = next_token_map.repeat(2, 1, 1)

    for b in var_model.blocks:
        b.attn.kv_caching(False)

    return var_model.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)


def load_models(device):
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=30, shared_aln=False,
        attn_l2_norm=True,
    )
    vae.load_state_dict(torch.load(VAE_CKPT, map_location="cpu"), strict=True)
    var.load_state_dict(torch.load(VAR_CKPT, map_location="cpu"), strict=True)
    vae.eval()
    var.eval()
    return vae, var


def main():
    parser = argparse.ArgumentParser(description="VAR img2img")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--class_label", type=int, required=True)
    parser.add_argument("--n_real_scales", type=int, default=5,
                        help="0=fully synthetic, 10=VQVAE recon")
    parser.add_argument("--output_dir", type=str, default="test_outputs")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=1.5)
    parser.add_argument("--top_k", type=int, default=900)
    parser.add_argument("--top_p", type=float, default=0.96)
    parser.add_argument("--sweep", action="store_true",
                        help="Test all n_real_scales from 0 to 10")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading models on GPU {args.gpu}...")
    vae, var = load_models(device)

    img_tensor = preprocess_image(args.image_path).to(device)
    orig_pil = tensor_to_pil(img_tensor.add(1).mul(0.5))

    if args.sweep:
        images = [orig_pil]
        labels = ["original"]
        for n in range(11):
            print(f"Generating n_real_scales={n}...")
            out = img2img_var(var, vae, img_tensor, args.class_label,
                              n_real_scales=n, cfg=args.cfg,
                              top_k=args.top_k, top_p=args.top_p, seed=args.seed)
            pil_img = tensor_to_pil(out)
            images.append(pil_img)
            labels.append(f"n={n}")
            pil_img.save(os.path.join(args.output_dir, f"var_img2img_n{n}.png"))

        # Save comparison grid
        w, h = images[0].size
        grid = Image.new("RGB", (w * len(images), h + 20), (255, 255, 255))
        for i, (img, lbl) in enumerate(zip(images, labels)):
            grid.paste(img, (i * w, 20))
        grid.save(os.path.join(args.output_dir, "var_img2img_sweep.png"))
        print(f"Sweep saved to {args.output_dir}/var_img2img_sweep.png")
    else:
        print(f"Generating with n_real_scales={args.n_real_scales}...")
        out = img2img_var(var, vae, img_tensor, args.class_label,
                          n_real_scales=args.n_real_scales, cfg=args.cfg,
                          top_k=args.top_k, top_p=args.top_p, seed=args.seed)
        pil_img = tensor_to_pil(out)
        out_path = os.path.join(args.output_dir,
                                f"var_img2img_n{args.n_real_scales}.png")
        pil_img.save(out_path)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
