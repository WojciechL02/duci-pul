#!/bin/bash
set -e
source /home/jdubinsk/dataset_inference_dm/.venv/bin/activate
W=/home/jdubinsk/dataset_inference_dm/synthetic/weights
mkdir -p "$W/VAR" "$W/RAR" "$W/Infinity"
echo "Downloading VAR-d30..."
python3 -c "from huggingface_hub import hf_hub_download as d; d('FoundationVision/var','var_d30.pth',local_dir='$W/VAR')"
echo "Downloading VAR VAE..."
python3 -c "from huggingface_hub import hf_hub_download as d; d('FoundationVision/var','vae_ch160v4096z32.pth',local_dir='$W/VAR')"
echo "Downloading RAR-XXL..."
python3 -c "from huggingface_hub import hf_hub_download as d; d('yucornetto/RAR','rar_xxl.bin',local_dir='$W/RAR')"
echo "Downloading MaskGIT tokenizer..."
python3 -c "from huggingface_hub import hf_hub_download as d; d('fun-research/TiTok','maskgit-vqgan-imagenet-f16-256.bin',local_dir='$W/RAR')"
echo "Downloading Infinity VAE..."
python3 -c "from huggingface_hub import hf_hub_download as d; d('FoundationVision/Infinity','infinity_vae_d32reg.pth',local_dir='$W/Infinity')"
echo "Downloading Infinity 2B..."
python3 -c "from huggingface_hub import hf_hub_download as d; d('FoundationVision/infinity','infinity_2b_reg.pth',local_dir='$W/Infinity')"
echo "ALL DOWNLOADS COMPLETE"
