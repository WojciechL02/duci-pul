#!/usr/bin/env python3
"""
Create a subset of ImageNet with N images per class.
Copies actual image files (no symlinks) so the subset is self-contained.
"""

import os
import argparse
import shutil
from pathlib import Path
import random


def create_subset(src_dir: str, dst_dir: str, images_per_class: int, seed: int = 42):
    """
    Create a subset of ImageNet with N images per class using actual file copies.
    
    Args:
        src_dir: Source ImageNet directory (e.g., /data/ImageNet100)
        dst_dir: Destination directory for subset
        images_per_class: Number of images to include per class
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    for split in ['train', 'val']:
        split_src = src_path / split
        split_dst = dst_path / split
        
        if not split_src.exists():
            print(f"Skipping {split} - source doesn't exist")
            continue
        
        print(f"Processing {split}...")
        
        # Get all class directories
        class_dirs = sorted([d for d in split_src.iterdir() if d.is_dir()])
        
        for class_dir in class_dirs:
            class_name = class_dir.name
            dst_class_dir = split_dst / class_name
            dst_class_dir.mkdir(parents=True, exist_ok=True)
            
            # Get all images in the class
            images = sorted([f for f in class_dir.iterdir() if f.is_file()])
            
            # Select N random images (or all if fewer than N)
            n_select = min(images_per_class, len(images))
            selected = random.sample(images, n_select)
            
            # Copy actual image files (no symlinks)
            for img in selected:
                dst_file = dst_class_dir / img.name
                if not dst_file.exists():
                    shutil.copy2(img, dst_file)
        
        # Count results
        total_images = sum(len(list((split_dst / d.name).iterdir())) 
                          for d in class_dirs if (split_dst / d.name).exists())
        print(f"  {split}: {len(class_dirs)} classes, {total_images} images")


def main():
    parser = argparse.ArgumentParser(description='Create ImageNet subset with N images per class')
    parser.add_argument('--src', type=str, default='/data/ImageNet100',
                        help='Source ImageNet directory')
    parser.add_argument('--dst', type=str, required=True,
                        help='Destination directory for subset')
    parser.add_argument('--images-per-class', type=int, default=5,
                        help='Number of images per class (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    print(f"Creating ImageNet subset:")
    print(f"  Source: {args.src}")
    print(f"  Destination: {args.dst}")
    print(f"  Images per class: {args.images_per_class}")
    print(f"  Seed: {args.seed}")
    print()
    
    create_subset(args.src, args.dst, args.images_per_class, args.seed)
    print("\nDone!")


if __name__ == '__main__':
    main()
