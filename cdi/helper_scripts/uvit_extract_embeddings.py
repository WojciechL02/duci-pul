#!/usr/bin/env python3
"""
Extract CLIP text embeddings for COCO captions.

This script extracts text embeddings from COCO captions using CLIP,
which are required for UViT text-to-image models.

Usage:
    python uvit_extract_embeddings.py --coco-dir /path/to/coco --split train
    python uvit_extract_embeddings.py --coco-dir /path/to/coco --split val
"""

import sys
import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path

# Fix sys.path so we use the local U-ViT and latent-diffusion modules
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
uvit_path = os.path.join(repo_root, "U-ViT")
ld_path = os.path.join(repo_root, "latent-diffusion")

# Insert at the front so it takes priority over site-packages
sys.path.insert(0, uvit_path)
sys.path.insert(0, ld_path)

print("Using paths:")
print(f"  U-ViT: {uvit_path}")
print(f"  latent-diffusion: {ld_path}")

# Now import from local modules
from datasets import MSCOCODatabase  # from U-ViT/datasets.py
import libs.autoencoder
import libs.clip


def main():
    parser = argparse.ArgumentParser(
        description='Extract CLIP text embeddings for COCO captions'
    )
    parser.add_argument(
        '--coco-dir', required=True,
        help='Path to COCO dataset directory'
    )
    parser.add_argument(
        '--split', default='train', choices=['train', 'val'],
        help='Dataset split to process (default: train)'
    )
    parser.add_argument(
        '--resolution', default=256, type=int,
        help='Image resolution (default: 256)'
    )
    args = parser.parse_args()
    
    coco_dir = Path(args.coco_dir)
    print(f"\nProcessing {args.split} split from {coco_dir}")

    # Setup paths based on split
    if args.split == "train":
        img_dir = coco_dir / "train2014"
        ann_file = coco_dir / "annotations" / "captions_train2014.json"
        save_dir = coco_dir / "train_text_emb"
    else:  # val
        img_dir = coco_dir / "val2014"
        ann_file = coco_dir / "annotations" / "captions_val2014.json"
        save_dir = coco_dir / "val_text_emb"
    
    # Verify paths exist
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not ann_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")
    
    # Create output directory
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving embeddings to: {save_dir}")
    
    # Load dataset
    datas = MSCOCODatabase(
        root=str(img_dir),
        annFile=str(ann_file),
        size=args.resolution
    )
    print(f"Dataset size: {len(datas)} images")

    # Setup CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    clip = libs.clip.FrozenCLIPEmbedder()
    clip.eval().to(device)

    # Extract embeddings
    print("\nExtracting embeddings...")
    with torch.no_grad():
        for idx, data in tqdm(enumerate(datas), total=len(datas)):
            _, captions = data
            latent = clip.encode(captions)
            for i, emb in enumerate(latent):
                c = emb.detach().cpu().numpy()
                np.save(save_dir / f"{idx}_{i}.npy", c)

    print(f"\nDone! Embeddings saved to: {save_dir}")
    print(f"Total embeddings: {len(datas) * 4} files (4 per image)")


if __name__ == "__main__":
    main()
