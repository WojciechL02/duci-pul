import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "VAR"))
import torch
from PIL import Image
from models import build_vae_var

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "VAR")
VAE_CKPT = os.path.join(WEIGHTS, "vae_ch160v4096z32.pth")
VAR_CKPT = os.path.join(WEIGHTS, "var_d30.pth")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_outputs")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=30, shared_aln=False,
        attn_l2_norm=True,
    )
    print("Loading VAE weights...")
    vae.load_state_dict(torch.load(VAE_CKPT, map_location="cpu"), strict=True)
    print("Loading VAR weights...")
    var.load_state_dict(torch.load(VAR_CKPT, map_location="cpu"), strict=True)
    vae.eval(); var.eval()
    print("Models loaded!")
    print("Generating class 207 (golden retriever)...")
    with torch.inference_mode():
        img = var.autoregressive_infer_cfg(
            B=1, label_B=207, g_seed=42, cfg=1.5,
            top_k=900, top_p=0.96, more_smooth=False,
        )
    img_np = (img[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    out_path = os.path.join(OUT_DIR, "var_test_class207.png")
    Image.fromarray(img_np).save(out_path)
    print(f"Saved to {out_path}, shape={img.shape}")
    print("VAR test PASSED!")

if __name__ == "__main__":
    main()
