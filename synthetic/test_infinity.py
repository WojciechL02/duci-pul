import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Infinity"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Infinity", "tools"))
import argparse
import torch
import numpy as np
from PIL import Image
from run_infinity import load_tokenizer, load_visual_tokenizer, load_transformer, gen_one_img
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "Infinity")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_outputs")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.cuda.set_device(0)
    
    args = argparse.Namespace(
        pn="0.06M",
        model_path=os.path.join(WEIGHTS, "infinity_2b_reg.pth"),
        cfg_insertion_layer=0,
        vae_type=32,
        vae_path=os.path.join(WEIGHTS, "infinity_vae_d32reg.pth"),
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        model_type="infinity_2b",
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0,
        sampling_per_bits=1,
        text_encoder_ckpt=os.path.join(WEIGHTS, "flan-t5-xl"),
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

    print("Loading text encoder...")
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    print("Loading VAE...")
    vae = load_visual_tokenizer(args)
    print("Loading Infinity transformer...")
    infinity = load_transformer(vae, args)
    
    h_div_w = 1.0
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]["scales"]
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

    prompt = "a golden retriever dog sitting in a garden, photorealistic"
    print(f"Generating: {prompt}")
    
    generated_image = gen_one_img(
        infinity, vae, text_tokenizer, text_encoder,
        prompt, g_seed=42, gt_leak=0, gt_ls_Bl=None,
        cfg_list=3, tau_list=0.5,
        scale_schedule=scale_schedule,
        cfg_insertion_layer=[args.cfg_insertion_layer],
        vae_type=args.vae_type,
        sampling_per_bits=args.sampling_per_bits,
        enable_positive_prompt=0,
    )
    
    img_np = generated_image.cpu().numpy()
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    if len(img_np.shape) == 3 and img_np.shape[0] == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
    # Convert BGR to RGB if needed (cv2 format)
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        img_np_rgb = img_np[:, :, ::-1] if img_np.shape[2] == 3 else img_np
    
    out_path = os.path.join(OUT_DIR, "infinity_test.png")
    Image.fromarray(img_np_rgb.astype(np.uint8)).save(out_path)
    print(f"Saved to {out_path}")
    print("Infinity test PASSED!")

if __name__ == "__main__":
    main()
