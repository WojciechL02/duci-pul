"""
Stable Diffusion img2img helper for synthetic dataset generation.

Loads SD VAE (sd-vae-ft-mse) and StableDiffusionImg2ImgPipeline (SD 1.5).
All HuggingFace downloads are cached under synthetic/StableDiffusionimg2Img.

Reconstructed from __pycache__/img2img_sd.cpython-312.pyc (the original file
was deleted); extended with a configurable working resolution `size` so the
same module supports 256 targets (VAR-d30 era) and 512 targets (VAR-d36).

Run (single-image smoke test):
  python img2img_sd.py --image_path /path/to/image.png --prompt "goldfish" --output_dir test_outputs
"""
import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SD_CACHE_DIR = os.path.join(SCRIPT_DIR, "StableDiffusionimg2Img")
os.makedirs(SD_CACHE_DIR, exist_ok=True)

os.environ.setdefault("HF_HOME", SD_CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(SD_CACHE_DIR, "transformers"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(SD_CACHE_DIR, "hub"))

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL, StableDiffusionImg2ImgPipeline

# runwayml/stable-diffusion-v1-5 was removed from the HF Hub; the official
# mirror hosts identical weights.
SD15_IDS = ("runwayml/stable-diffusion-v1-5", "stable-diffusion-v1-5/stable-diffusion-v1-5")


def load_models(device):
    """Load SD VAE and img2img pipeline. Downloads go to StableDiffusionimg2Img/."""
    ae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    dtype = (
        torch.float16
        if isinstance(device, torch.device) and device.type == "cuda"
        else torch.float32
    )
    last_err = None
    for model_id in SD15_IDS:
        try:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id, torch_dtype=dtype, safety_checker=None
            ).to(device)
            break
        except Exception as e:  # repo gated/removed -> try mirror
            last_err = e
    else:
        raise last_err
    return ae, pipe


@torch.no_grad()
def batch_ae_reconstruct(ae, imgs_01, device):
    """VAE encode + decode. imgs_01: [B,3,H,W] in [0,1], output [B,3,H,W] in [0,1]."""
    x_norm = imgs_01.to(device) * 2 - 1
    lat = ae.encode(x_norm).latent_dist.sample()
    rec = (ae.decode(lat).sample + 1) * 0.5
    return rec.cpu()


@torch.no_grad()
def img2img_sd(pipe, ae, image_input, prompt, strength=0.5, guidance_scale=7.5,
               seed=42, device=None, size=256):
    """
    Single-image img2img. image_input: PIL Image or [1,3,size,size] tensor in [0,1].
    SD 1.5 always runs at 512x512 internally; output is resized to `size`.
    Returns PIL Image at size x size.
    """
    if device is None:
        device = next(pipe.unet.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed)

    if isinstance(image_input, torch.Tensor):
        pil_512 = transforms.ToPILImage()(image_input[0].cpu()).resize(
            (512, 512), Image.BICUBIC
        )
    else:
        pil_512 = (
            image_input.resize((512, 512), Image.BICUBIC)
            if image_input.size != (512, 512)
            else image_input
        )

    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        out = pipe(
            prompt=prompt,
            image=pil_512,
            strength=strength,
            guidance_scale=guidance_scale,
            generator=gen,
        )
    out_pil = out.images[0]
    if out_pil.size != (size, size):
        out_pil = out_pil.resize((size, size), Image.BICUBIC)
    return out_pil


@torch.no_grad()
def batch_img2img_sd(pipe, imgs_01, prompts, strength=0.5, guidance_scale=7.5,
                     seed_base=42, device=None, size=256):
    """
    Batched img2img. imgs_01: [B,3,size,size] in [0,1]. prompts: list of B strings.
    SD 1.5 always runs at 512x512 internally; outputs are resized to `size`.
    Returns list of PIL Images size x size.
    """
    if device is None:
        device = next(pipe.unet.parameters()).device
    B = imgs_01.shape[0]
    to_pil = transforms.ToPILImage()
    pil_512_list = []
    for i in range(B):
        pil = to_pil(imgs_01[i].cpu())
        if pil.size != (512, 512):
            pil = pil.resize((512, 512), Image.BICUBIC)
        pil_512_list.append(pil)
    gen = torch.Generator(device=device).manual_seed(seed_base)
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        out = pipe(
            prompt=prompts,
            image=pil_512_list,
            strength=strength,
            guidance_scale=guidance_scale,
            generator=gen,
        )
    return [
        im if im.size == (size, size) else im.resize((size, size), Image.BICUBIC)
        for im in out.images
    ]


def preprocess_image(image_path, size=256):
    """Load image, resize, center crop, ToTensor [0,1]."""
    img = Image.open(image_path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])
    return tf(img).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion img2img (single image)")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="test_outputs")
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--no_ae", action="store_true", help="Do not save AE reconstruction")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading SD VAE and img2img pipeline…")
    ae, pipe = load_models(device)
    print("Models ready.")

    img_tensor = preprocess_image(args.image_path, size=args.size).to(device)
    orig_pil = Image.fromarray(
        (img_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    )
    orig_pil.save(os.path.join(args.output_dir, f"original_{args.size}.png"))

    if not args.no_ae:
        rec = batch_ae_reconstruct(ae, img_tensor, device)
        rec_pil = Image.fromarray(
            (rec[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
        )
        rec_pil.save(os.path.join(args.output_dir, "ae_recon.png"))
        print("Saved AE reconstruction to ae_recon.png")

    i2i_pil = img2img_sd(
        pipe, ae, img_tensor, args.prompt, strength=args.strength,
        guidance_scale=args.guidance_scale, seed=args.seed, device=device,
        size=args.size,
    )
    i2i_pil.save(os.path.join(args.output_dir, "img2img_sd.png"))
    print(f"Saved img2img to img2img_sd.png (prompt: {args.prompt[:50]}…)")


if __name__ == "__main__":
    main()
