#!/bin/bash
# Download COCO 2014 dataset for UViT text-to-image models
# Usage: ./download_coco.sh ../data/coco

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <output_directory>"
    echo "Example: $0 ../data/coco"
    exit 1
fi

COCO_DIR="$1"
mkdir -p "$COCO_DIR"
cd "$COCO_DIR"

echo "=== Downloading COCO 2014 dataset to $COCO_DIR ==="

# Download train images (~13GB)
if [ ! -f "train2014.zip" ] && [ ! -d "train2014" ]; then
    echo "[1/3] Downloading train2014 images..."
    wget -c http://images.cocodataset.org/zips/train2014.zip
else
    echo "[1/3] train2014 already exists, skipping download"
fi

# Download val images (~6GB)
if [ ! -f "val2014.zip" ] && [ ! -d "val2014" ]; then
    echo "[2/3] Downloading val2014 images..."
    wget -c http://images.cocodataset.org/zips/val2014.zip
else
    echo "[2/3] val2014 already exists, skipping download"
fi

# Download annotations
if [ ! -f "annotations_trainval2014.zip" ] && [ ! -d "annotations" ]; then
    echo "[3/3] Downloading annotations..."
    wget -c http://images.cocodataset.org/annotations/annotations_trainval2014.zip
else
    echo "[3/3] annotations already exist, skipping download"
fi

# Extract files
echo "=== Extracting files ==="

if [ -f "train2014.zip" ]; then
    echo "Extracting train2014.zip..."
    unzip -q train2014.zip
    rm train2014.zip
    echo "train2014 extracted"
fi

if [ -f "val2014.zip" ]; then
    echo "Extracting val2014.zip..."
    unzip -q val2014.zip
    rm val2014.zip
    echo "val2014 extracted"
fi

if [ -f "annotations_trainval2014.zip" ]; then
    echo "Extracting annotations..."
    unzip -q annotations_trainval2014.zip
    rm annotations_trainval2014.zip
    echo "annotations extracted"
fi

echo ""
echo "=== COCO 2014 download complete ==="
echo "Directory structure:"
ls -la "$COCO_DIR"
echo ""
echo "Next steps:"
echo "1. Extract text embeddings using:"
echo "   cd .."
echo "   source .venv/bin/activate"
echo "   python helper_scripts/uvit_extract_embeddings.py --split train"
echo "   python helper_scripts/uvit_extract_embeddings.py --split val"
