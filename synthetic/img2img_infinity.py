"""
img2img for Infinity (Bitwise Visual AutoRegressive Modeling).

Encodes a real image through Infinity's BSQ-VAE to get multi-scale
bit labels, then uses the built-in gt_leak mechanism to keep the first
n_real_scales from the real image and generate the rest.

With pn='0.06M' (256x256), Infinity uses 7 scales.
gt_leak=0 -> fully synthetic, gt_leak=K -> VAE reconstruction.
"""
import argparse
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "Infinity"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "Infinity", "tools"))

import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor

from run_infinity import (
    load_tokenizer, load_visual_tokenizer, load_transformer,
    gen_one_img, encode_prompt, transform,
)
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates

WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights", "Infinity")
CACHE_DIR = os.environ.get("INFINITY_CACHE_DIR", os.path.join(SCRIPT_DIR, ".cache"))


def get_infinity_args(gpu=1, pn="0.06M"):
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


def encode_image(vae, image_path, scale_schedule, device, tgt_h, tgt_w):
    """Encode a real image through Infinity's BSQ-VAE."""
    pil_image = Image.open(image_path).convert("RGB")
    inp = transform(pil_image, tgt_h, tgt_w)
    inp = inp.unsqueeze(0).to(device)
    schedule = [(item[0], item[1], item[2]) for item in scale_schedule]
    h, z, _, all_bit_indices, _, _ = vae.encode(inp, scale_schedule=schedule)
    recons_img = vae.decode(z)[0]
    if len(recons_img.shape) == 4:
        recons_img = recons_img.squeeze(1)
    recons_np = ((recons_img + 1) / 2).permute(1, 2, 0).mul_(255).cpu().numpy().astype(np.uint8)
    input_np = ((inp[0] + 1) / 2).permute(1, 2, 0).mul_(255).cpu().numpy().astype(np.uint8)
    return input_np, recons_np, all_bit_indices


def main():
    parser = argparse.ArgumentParser(description="Infinity img2img")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="a photo")
    parser.add_argument("--gt_leak", type=int, default=3,
                        help="Number of scales from real image (0=synthetic, K=recon)")
    parser.add_argument("--output_dir", type=str, default="test_outputs")
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--pn", type=str, default="0.06M",
                        choices=["0.06M", "0.25M", "1M"])
    parser.add_argument("--sweep", action="store_true",
                        help="Test all gt_leak values from 0 to K")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.makedirs(args.output_dir, exist_ok=True)

    torch.cuda.set_device(0)
    inf_args = get_infinity_args(args.gpu, args.pn)

    print("Loading text encoder...")
    text_tokenizer, text_encoder = load_tokenizer(t5_path=inf_args.text_encoder_ckpt)
    print("Loading VAE...")
    vae = load_visual_tokenizer(inf_args)
    print("Loading Infinity transformer...")
    infinity = load_transformer(vae, inf_args)

    h_div_w = 1.0
    h_div_w_template_ = h_div_w_templates[
        np.argmin(np.abs(h_div_w_templates - h_div_w))
    ]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]["scales"]
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    K = len(scale_schedule)

    # Determine target resolution from the last scale
    tgt_h = scale_schedule[-1][1] * 16
    tgt_w = scale_schedule[-1][2] * 16
    if inf_args.apply_spatial_patchify:
        tgt_h = scale_schedule[-1][1] * 8
        tgt_w = scale_schedule[-1][2] * 8

    print(f"Scale schedule has {K} scales, target resolution: {tgt_h}x{tgt_w}")

    # Encode real image
    print(f"Encoding real image: {args.image_path}")
    input_np, recon_np, gt_ls_Bl = encode_image(
        vae, args.image_path, scale_schedule, "cuda", tgt_h, tgt_w
    )

    if args.sweep:
        images = [Image.fromarray(input_np), Image.fromarray(recon_np)]
        labels = ["input", "vae_recon"]

        for gl in range(K + 1):
            print(f"Generating gt_leak={gl}/{K}...")
            generated = gen_one_img(
                infinity, vae, text_tokenizer, text_encoder,
                args.prompt,
                g_seed=args.seed,
                gt_leak=gl,
                gt_ls_Bl=gt_ls_Bl,
                cfg_list=args.cfg,
                tau_list=args.tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[inf_args.cfg_insertion_layer],
                vae_type=inf_args.vae_type,
                sampling_per_bits=inf_args.sampling_per_bits,
                enable_positive_prompt=0,
            )
            img_np = generated.cpu().numpy()
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            if len(img_np.shape) == 3 and img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            # Infinity returns BGR, convert to RGB
            img_np_rgb = img_np[:, :, ::-1].copy()
            pil_img = Image.fromarray(img_np_rgb.astype(np.uint8))
            images.append(pil_img)
            labels.append(f"gl={gl}")
            pil_img.save(os.path.join(args.output_dir, f"infinity_img2img_gl{gl}.png"))

        # Save comparison grid
        w, h = images[0].size
        grid = Image.new("RGB", (w * len(images), h + 20), (255, 255, 255))
        for i, img in enumerate(images):
            if img.size != (w, h):
                img = img.resize((w, h), Image.LANCZOS)
            grid.paste(img, (i * w, 20))
        grid.save(os.path.join(args.output_dir, "infinity_img2img_sweep.png"))
        print(f"Sweep saved to {args.output_dir}/infinity_img2img_sweep.png")
    else:
        print(f"Generating with gt_leak={args.gt_leak}/{K}...")
        generated = gen_one_img(
            infinity, vae, text_tokenizer, text_encoder,
            args.prompt,
            g_seed=args.seed,
            gt_leak=args.gt_leak,
            gt_ls_Bl=gt_ls_Bl,
            cfg_list=args.cfg,
            tau_list=args.tau,
            scale_schedule=scale_schedule,
            cfg_insertion_layer=[inf_args.cfg_insertion_layer],
            vae_type=inf_args.vae_type,
            sampling_per_bits=inf_args.sampling_per_bits,
            enable_positive_prompt=0,
        )
        img_np = generated.cpu().numpy()
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        if len(img_np.shape) == 3 and img_np.shape[0] == 3:
            img_np = np.transpose(img_np, (1, 2, 0))
        img_np_rgb = img_np[:, :, ::-1].copy()
        pil_img = Image.fromarray(img_np_rgb.astype(np.uint8))
        out_path = os.path.join(args.output_dir, f"infinity_img2img_gl{args.gt_leak}.png")
        pil_img.save(out_path)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
