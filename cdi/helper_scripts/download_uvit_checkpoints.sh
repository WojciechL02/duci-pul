#!/bin/bash
# Download UViT pretrained model checkpoints
#
# The checkpoints need to be downloaded from Google Drive.
# This script provides the links and creates the directory structure.
#
# Usage: ./download_uvit_checkpoints.sh

set -e

CHECKPOINT_DIR="./model_checkpoints/uvit_t2i"
mkdir -p "$CHECKPOINT_DIR"

echo "=== UViT Model Checkpoints ==="
echo ""
echo "The following files need to be downloaded from Google Drive:"
echo ""
echo "1. MS-COCO U-ViT-S/2 (mscoco_uvit_small.pth):"
echo "   https://drive.google.com/file/d/15JsZWRz2byYNU6K093et5e5Xqd4uwA8S/view"
echo ""
echo "2. MS-COCO U-ViT-S/2 Deep (mscoco_uvit_small_deep.pth):"
echo "   https://drive.google.com/file/d/1gHRy8sn039Wy-iFL21wH8TiheHK8Ky71/view"
echo ""
echo "3. Autoencoder (autoencoder_kl.pth) from stable-diffusion folder:"
echo "   https://drive.google.com/drive/folders/1yo-XhqbPue3rp5P57j6QbA5QZx6KybvP"
echo "   Download the 'autoencoder_kl.pth' file from inside the stable-diffusion folder"
echo ""
echo "Download these files and place them in: $CHECKPOINT_DIR"
echo ""

# Check for gdown (pip install gdown)
if command -v gdown; then
    echo "gdown found. Attempting automatic download..."
    echo ""
    
    cd "$CHECKPOINT_DIR"
    
    if [ ! -f "mscoco_uvit_small.pth" ]; then
        echo "Downloading mscoco_uvit_small.pth..."
        gdown "15JsZWRz2byYNU6K093et5e5Xqd4uwA8S" -O mscoco_uvit_small.pth
    else
        echo "mscoco_uvit_small.pth already exists"
    fi
    
    if [ ! -f "mscoco_uvit_small_deep.pth" ]; then
        echo "Downloading mscoco_uvit_small_deep.pth..."
        gdown "1gHRy8sn039Wy-iFL21wH8TiheHK8Ky71" -O mscoco_uvit_small_deep.pth
    else
        echo "mscoco_uvit_small_deep.pth already exists"
    fi
    
    # Note: autoencoder needs manual download as it's in a shared folder
    if [ ! -f "autoencoder_kl.pth" ]; then
        echo ""
        echo "NOTE: autoencoder_kl.pth needs manual download from the Google Drive folder."
        echo "Go to: https://drive.google.com/drive/folders/1yo-XhqbPue3rp5P57j6QbA5QZx6KybvP"
        echo "Navigate to stable-diffusion folder and download autoencoder_kl.pth"
        echo "Then move it to: $CHECKPOINT_DIR/autoencoder_kl.pth"
    else
        echo "autoencoder_kl.pth already exists"
    fi
    
    cd -
else
    echo "gdown not found. Install it with: pip install gdown"
    echo "Or download the files manually from the links above."
fi

echo ""
echo "=== Verification ==="
echo "Expected files in $CHECKPOINT_DIR:"
echo "  - mscoco_uvit_small.pth"
echo "  - mscoco_uvit_small_deep.pth"  
echo "  - autoencoder_kl.pth"
echo ""
echo "Current contents:"
ls -la "$CHECKPOINT_DIR" || echo "(directory not found)"
