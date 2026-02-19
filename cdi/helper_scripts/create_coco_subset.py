#!/usr/bin/env python3
"""
Create a subset of COCO dataset with a specified number of images.

This script copies actual image files (no symlinks) and the corresponding
text embeddings. Subset annotations refer only to images that exist in the subset.

Usage:
    python create_coco_subset.py --src /path/to/coco --dst /path/to/coco_5000 --num-train 5000 --num-val 5000
"""

import os
import argparse
import json
import shutil
import random
from pathlib import Path
from typing import List, Dict, Any


def create_subset(
    src_dir: str,
    dst_dir: str,
    num_train: int,
    num_val: int,
    seed: int = 42
):
    """
    Create a COCO subset with actual image copies and copied embeddings.
    Annotations written refer only to the selected images in the subset.
    
    Args:
        src_dir: Path to source COCO directory
        dst_dir: Path to destination subset directory
        num_train: Number of training images to include
        num_val: Number of validation images to include
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    # Process each split
    for split, num_images in [('train', num_train), ('val', num_val)]:
        print(f"\nProcessing {split}...")
        
        img_folder = f"{split}2014"
        emb_folder = f"{split}_text_emb"
        ann_file = f"captions_{split}2014.json"
        
        # Check source directories exist
        src_img_dir = src_path / img_folder
        src_emb_dir = src_path / emb_folder
        src_ann_file = src_path / "annotations" / ann_file
        
        if not src_img_dir.exists():
            print(f"  Warning: {src_img_dir} not found, skipping {split}")
            continue
        if not src_ann_file.exists():
            print(f"  Warning: {src_ann_file} not found, skipping {split}")
            continue
        
        # Load annotations
        with open(src_ann_file, 'r') as f:
            ann_data = json.load(f)
        
        # Get list of images and sample
        all_images: List[Dict[str, Any]] = ann_data['images']
        if num_images < len(all_images):
            selected_images = random.sample(all_images, num_images)
        else:
            selected_images = all_images
            print(f"  Warning: Requested {num_images} but only {len(all_images)} available")
        
        selected_ids = {img['id'] for img in selected_images}
        
        # Create destination directories
        dst_img_dir = dst_path / img_folder
        dst_emb_dir = dst_path / emb_folder
        dst_ann_dir = dst_path / "annotations"
        
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_emb_dir.mkdir(parents=True, exist_ok=True)
        dst_ann_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy actual image files (no symlinks)
        for img in selected_images:
            src_img = src_img_dir / img['file_name']
            dst_img = dst_img_dir / img['file_name']
            if src_img.exists() and not dst_img.exists():
                shutil.copy2(src_img, dst_img)
        
        # Copy text embeddings - need to find the index mapping
        # The embeddings are stored as {index}_{k}.npy where index is the position
        # in the sorted image keys list
        
        # Build original index mapping
        original_keys = sorted([img['id'] for img in ann_data['images']])
        original_idx_map = {key: idx for idx, key in enumerate(original_keys)}
        
        # Build new index mapping for selected images
        selected_keys = sorted(list(selected_ids))
        new_idx_map = {key: idx for idx, key in enumerate(selected_keys)}
        
        # Copy embeddings with new indices
        if src_emb_dir.exists():
            for key in selected_keys:
                old_idx = original_idx_map[key]
                new_idx = new_idx_map[key]
                
                # Copy all 4 embedding variants (k=0,1,2,3)
                for k in range(4):
                    src_emb = src_emb_dir / f"{old_idx}_{k}.npy"
                    dst_emb = dst_emb_dir / f"{new_idx}_{k}.npy"
                    if src_emb.exists() and not dst_emb.exists():
                        shutil.copy2(src_emb, dst_emb)
        else:
            print(f"  Warning: Embeddings not found at {src_emb_dir}")
        
        # Create filtered annotations
        filtered_annotations = [
            ann for ann in ann_data['annotations']
            if ann['image_id'] in selected_ids
        ]
        
        new_ann_data = {
            'info': ann_data.get('info', {}),
            'licenses': ann_data.get('licenses', []),
            'images': selected_images,
            'annotations': filtered_annotations
        }
        
        dst_ann_file = dst_ann_dir / ann_file
        with open(dst_ann_file, 'w') as f:
            json.dump(new_ann_data, f)
        
        print(f"  {split}: {len(selected_images)} images, {len(filtered_annotations)} annotations")
    
    print(f"\nDone! Subset created at: {dst_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Create a subset of COCO dataset'
    )
    parser.add_argument(
        '--src', required=True,
        help='Source COCO directory'
    )
    parser.add_argument(
        '--dst', required=True,
        help='Destination subset directory'
    )
    parser.add_argument(
        '--num-train', type=int, default=5000,
        help='Number of training images (default: 5000)'
    )
    parser.add_argument(
        '--num-val', type=int, default=5000,
        help='Number of validation images (default: 5000)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    create_subset(args.src, args.dst, args.num_train, args.num_val, args.seed)


if __name__ == '__main__':
    main()
