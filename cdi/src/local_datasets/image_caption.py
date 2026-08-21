"""Image + caption dataset that walks an ImageFolder-like layout produced by
`cdi/bash_scripts/prep_cc12m_for_mia.py`.

Directory layout expected under ``dataset_path / split``:
    <root>/<subdir>/<uid>.jpg   (image)
    <root>/<subdir>/<uid>.txt   (caption)

This is used by the Nitro-T T2I wrapper's CLiD run where we need (image, caption)
pairs instead of (image, class_id) like the ImageNet-style loader.
"""
from __future__ import annotations

import glob
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose


def _t2i_collate(batch):
    import torch

    images = torch.stack([b[0] for b in batch], dim=0)
    captions = [b[1] for b in batch]
    paths = [b[2] for b in batch]
    return images, captions, paths


class ImageCaptionDataset(Dataset):
    def __init__(self, dataset_cfg, transform: Compose):
        self.root = os.path.join(dataset_cfg.dataset_path, dataset_cfg.split)
        self.transform = transform
        patterns = [os.path.join(self.root, "**", "*.jpg"),
                    os.path.join(self.root, "**", "*.jpeg"),
                    os.path.join(self.root, "**", "*.png")]
        files: list[str] = []
        for p in patterns:
            files.extend(glob.glob(p, recursive=True))
        files.sort()
        if not files:
            raise FileNotFoundError(f"No images under {self.root}")
        self._files = files

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx: int):
        img_path = self._files[idx]
        cap_path = os.path.splitext(img_path)[0] + ".txt"
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        caption = ""
        if os.path.exists(cap_path):
            with open(cap_path) as fh:
                caption = fh.read().strip()
        return image, caption, os.path.abspath(img_path)


class ImageCaptionRegistry:
    """Tiny wrapper matching the ImageFolderDataset API (`.dataset`, `.collate_fn`)."""
    def __init__(self, dataset_cfg, transform: Compose):
        self.dataset = ImageCaptionDataset(dataset_cfg, transform)
        self.collate_fn = _t2i_collate
