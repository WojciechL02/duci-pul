import os
import json
import random
import shutil
from tqdm import tqdm

# === CONFIG ===
NUM_IMAGES = 5000
RANDOM_SEED = 42

COCO_ROOT = './'  # path to your COCO root
ANN_DIR = os.path.join(COCO_ROOT, 'annotations')

# You already have reduced images here:
TRAIN_IMG_DIR = os.path.join(COCO_ROOT, 'train2014')
VAL_IMG_DIR = os.path.join(COCO_ROOT, 'val2014')

random.seed(RANDOM_SEED)

def reduce_captions(split_name):
    ann_path = os.path.join(ANN_DIR, f'captions_{split_name}2014.json')
    out_path = ann_path  # overwrite in place

    print(f"\n=== Reducing {ann_path} ===")

    with open(ann_path, 'r') as f:
        data = json.load(f)

    # Collect image IDs actually present in the folder
    img_dir = TRAIN_IMG_DIR if split_name == 'train' else VAL_IMG_DIR
    existing_files = {f for f in os.listdir(img_dir) if f.endswith('.jpg')}

    # Filter images to only those existing
    reduced_images = [img for img in data['images'] if img['file_name'] in existing_files]
    existing_ids = {img['id'] for img in reduced_images}

    # Filter annotations (captions) to only existing image_ids
    reduced_annotations = [ann for ann in data['annotations'] if ann['image_id'] in existing_ids]

    reduced_data = {
        'info': data.get('info', {}),
        'licenses': data.get('licenses', []),
        'images': reduced_images,
        'annotations': reduced_annotations
    }

    with open(out_path, 'w') as f:
        json.dump(reduced_data, f)

    print(f"✅ Saved reduced captions file → {out_path}")
    print(f"   {len(reduced_images)} images, {len(reduced_annotations)} captions kept.")

reduce_captions('train')
reduce_captions('val')
print("\n🎉 Done! Your captions JSONs now match the reduced 5k image subsets.")
