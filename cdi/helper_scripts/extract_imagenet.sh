#!/bin/bash
# Extract and organize ImageNet-1K after torrent download

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMAGENET_DIR="${1:-$REPO_ROOT/data/imagenet_1k}"

echo "Extracting ImageNet to: $IMAGENET_DIR"
cd "$IMAGENET_DIR"

# Extract training data
if [ -f "ILSVRC2012_img_train.tar" ]; then
    echo "Extracting training data..."
    mkdir -p train
    tar -xf ILSVRC2012_img_train.tar -C train/
    
    echo "Extracting class folders..."
    cd train
    for f in n*.tar; do
        if [ -f "$f" ]; then
            class="${f%.tar}"
            mkdir -p "$class"
            tar -xf "$f" -C "$class"
            rm "$f"
            echo -ne "\r  Extracted: $class"
        fi
    done
    echo -e "\nTraining data extracted!"
    cd ..
else
    echo "ILSVRC2012_img_train.tar not found, skipping..."
fi

# Extract validation data
if [ -f "ILSVRC2012_img_val.tar" ]; then
    echo "Extracting validation data..."
    mkdir -p val
    tar -xf ILSVRC2012_img_val.tar -C val/
    
    echo "Organizing validation images into class folders..."
    cd val
    
    # Download and run valprep.sh to organize val images
    wget -q https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh -O valprep.sh
    chmod +x valprep.sh
    ./valprep.sh
    rm valprep.sh
    
    echo "Validation data extracted and organized!"
    cd ..
else
    echo "ILSVRC2012_img_val.tar not found, skipping..."
fi

echo ""
echo "ImageNet extraction complete!"
echo "Directory structure:"
echo "  $IMAGENET_DIR/train/ - 1000 class folders"
echo "  $IMAGENET_DIR/val/   - 1000 class folders"

# Count
echo ""
echo "Statistics:"
echo "  Train classes: $(ls train/ | wc -l)"
echo "  Val classes: $(ls val/ | wc -l)"
echo "  Train images: $(find train/ -name '*.JPEG' | wc -l)"
echo "  Val images: $(find val/ -name '*.JPEG' | wc -l)"
