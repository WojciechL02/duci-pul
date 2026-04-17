"""
img2img for RAR (Randomized Autoregressive Visual Generation).

Encodes a real image to 256 discrete tokens using MaskGIT-VQGAN, then
during AR generation, at each step with probability `replace_ratio`
keeps the real image's token instead of the model's sampled token.

replace_ratio=0.0 -> fully synthetic, replace_ratio=1.0 -> original tokens.
"""
import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "RAR"))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

import demo_util
from utils.train_utils import create_pretrained_tokenizer

WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights", "RAR")


def preprocess_image(image_path, size=256):
    """MaskGIT-VQGAN expects [0, 1] range."""
    img = Image.open(image_path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])
    return tf(img).unsqueeze(0)


@torch.no_grad()
def generate_img2img(generator, real_tokens, condition, replace_ratio,
                     guidance_scale, randomize_temperature, guidance_scale_pow,
                     device):
    """
    Modified RAR generate that mixes real and generated tokens.

    At each AR step, after sampling from the model, with probability
    `replace_ratio` the sampled token is replaced with the real image's
    token at that position.
    """
    condition = generator.preprocess_condition(condition, cond_drop_prob=0.0)
    num_samples = condition.shape[0]
    ids = torch.full((num_samples, 0), -1, device=device, dtype=torch.long)

    generator.enable_kv_cache()

    for step in range(generator.image_seq_len):
        scale_pow = torch.ones((1,), device=device) * guidance_scale_pow
        scale_step = (1 - torch.cos(
            ((step / generator.image_seq_len) ** scale_pow) * torch.pi)) * 0.5
        cfg_scale = (guidance_scale - 1) * scale_step + 1

        if guidance_scale != 0:
            logits = generator.forward_fn(
                torch.cat([ids, ids], dim=0),
                torch.cat([condition, generator.get_none_condition(condition)], dim=0),
                orders=None, is_sampling=True,
            )
            cond_logits, uncond_logits = logits[:num_samples], logits[num_samples:]
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        else:
            logits = generator.forward_fn(ids, condition, orders=None, is_sampling=True)

        logits = logits[:, -1]
        logits = logits / randomize_temperature
        probs = F.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)

        # Teacher forcing: replace with real token based on replace_ratio
        use_real = torch.rand(num_samples, 1, device=device) < replace_ratio
        real_token_step = real_tokens[:, step : step + 1]
        token = torch.where(use_real, real_token_step, sampled)

        ids = torch.cat((ids, token), dim=-1)

    generator.disable_kv_cache()
    return ids


def load_models(device):
    rar_dir = os.path.join(SCRIPT_DIR, "RAR")
    config = demo_util.get_config(
        os.path.join(rar_dir, "configs", "training", "generator", "rar.yaml")
    )
    config.experiment.generator_checkpoint = os.path.join(WEIGHTS_DIR, "rar_xxl.bin")
    config.model.vq_model.pretrained_tokenizer_weight = os.path.join(
        WEIGHTS_DIR, "maskgit-vqgan-imagenet-f16-256.bin"
    )
    config.model.generator.hidden_size = 1408
    config.model.generator.num_hidden_layers = 40
    config.model.generator.num_attention_heads = 16
    config.model.generator.intermediate_size = 6144

    tokenizer = create_pretrained_tokenizer(config)
    tokenizer.to(device)
    generator = demo_util.get_rar_generator(config)
    generator.to(device)
    return tokenizer, generator


def main():
    parser = argparse.ArgumentParser(description="RAR img2img")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--class_label", type=int, required=True)
    parser.add_argument("--replace_ratio", type=float, default=0.8,
                        help="0.0=fully synthetic, 1.0=original tokens")
    parser.add_argument("--output_dir", type=str, default="test_outputs")
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_scale", type=float, default=8.0)
    parser.add_argument("--temperature", type=float, default=1.02)
    parser.add_argument("--guidance_pow", type=float, default=1.2)
    parser.add_argument("--sweep", action="store_true",
                        help="Test replace_ratios 0.0, 0.2, 0.4, 0.6, 0.8, 1.0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    print(f"Loading models on GPU {args.gpu}...")
    tokenizer, generator = load_models(device)

    img_tensor = preprocess_image(args.image_path).to(device)

    # Encode real image to tokens [B, 256] (16x16 grid flattened)
    real_tokens = tokenizer.encode(img_tensor)
    if real_tokens.dim() == 3:
        real_tokens = real_tokens.view(real_tokens.shape[0], -1)
    print(f"Encoded real image to {real_tokens.shape[1]} tokens")

    # Reconstruct original from tokens for reference
    recon = tokenizer.decode_tokens(real_tokens)
    orig_pil = Image.fromarray(
        (torch.clamp(recon, 0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    )

    # Also save the original input
    input_pil = Image.fromarray(
        (img_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    )

    if args.sweep:
        ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        images = [input_pil, orig_pil]
        labels = ["input", "vqvae_recon"]
        for r in ratios:
            print(f"Generating replace_ratio={r}...")
            condition = torch.LongTensor([args.class_label]).to(device)
            gen_tokens = generate_img2img(
                generator, real_tokens, condition, r,
                args.guidance_scale, args.temperature, args.guidance_pow, device,
            )
            gen_img = tokenizer.decode_tokens(gen_tokens)
            pil_img = Image.fromarray(
                (torch.clamp(gen_img, 0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            )
            images.append(pil_img)
            labels.append(f"r={r}")
            pil_img.save(os.path.join(args.output_dir, f"rar_img2img_r{r:.1f}.png"))

        # Save comparison grid
        w, h = images[0].size
        grid = Image.new("RGB", (w * len(images), h + 20), (255, 255, 255))
        for i, img in enumerate(images):
            grid.paste(img, (i * w, 20))
        grid.save(os.path.join(args.output_dir, "rar_img2img_sweep.png"))
        print(f"Sweep saved to {args.output_dir}/rar_img2img_sweep.png")
    else:
        print(f"Generating with replace_ratio={args.replace_ratio}...")
        condition = torch.LongTensor([args.class_label]).to(device)
        gen_tokens = generate_img2img(
            generator, real_tokens, condition, args.replace_ratio,
            args.guidance_scale, args.temperature, args.guidance_pow, device,
        )
        gen_img = tokenizer.decode_tokens(gen_tokens)
        pil_img = Image.fromarray(
            (torch.clamp(gen_img, 0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        )
        out_path = os.path.join(
            args.output_dir, f"rar_img2img_r{args.replace_ratio:.1f}.png"
        )
        pil_img.save(out_path)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
