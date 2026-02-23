"""
COCO dataloader for UViT text-to-image models.

Loads images from {dataset_path}/{split}2014/ and pre-extracted CLIP text
embeddings from {dataset_path}/{split}_text_emb/{idx}_{caption}.npy.

The embedding index is the position in sorted(coco.imgs.keys()), matching
the convention used by helper_scripts/uvit_extract_embeddings.py.

Returns batches of (image_tensor, text_embedding_tensor) where
text_embedding_tensor has shape [B, 77, 768].
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from omegaconf import DictConfig
from PIL import Image


def center_crop(width, height, img_array):
    """Center crop numpy array to (height, width)."""
    crop_h = img_array.shape[0]
    crop_w = img_array.shape[1]
    start_h = (crop_h - height) // 2
    start_w = (crop_w - width) // 2
    return img_array[start_h:start_h + height, start_w:start_w + width]


class COCOWithEmbeddings(Dataset):
    """COCO dataset that returns (image, text_embedding) pairs.

    Images are loaded from {img_dir}/ and text embeddings from {emb_dir}/.
    The annotations JSON determines the image order and the index-to-filename
    mapping that aligns with the embedding file naming convention.
    """

    def __init__(self, dataset_cfg, transform):
        split = dataset_cfg.split
        self.dataset_path = dataset_cfg.dataset_path
        self.img_dir = os.path.join(self.dataset_path, f"{split}2014")
        self.emb_dir = os.path.join(self.dataset_path, f"{split}_text_emb")
        self.ann_file = os.path.join(
            self.dataset_path, "annotations", f"captions_{split}2014.json")
        self.transform = transform

        from pycocotools.coco import COCO
        self.coco = COCO(self.ann_file)
        self.keys = list(sorted(self.coco.imgs.keys()))

        self.n_captions = self._count_captions()

    def _count_captions(self):
        """Detect number of caption embeddings per image from files on disk."""
        files = glob.glob(os.path.join(self.emb_dir, "0_*.npy"))
        return max(len(files), 1)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        info = self.coco.loadImgs(key)[0]
        fname = info["file_name"]

        img_path = os.path.join(self.img_dir, fname)
        if not os.path.exists(img_path) and fname.endswith(".jpg"):
            img_path = os.path.join(self.img_dir, fname[:-4] + ".png")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        caption_idx = np.random.randint(0, self.n_captions)
        emb_path = os.path.join(self.emb_dir, f"{index}_{caption_idx}.npy")
        if os.path.exists(emb_path):
            emb = torch.from_numpy(np.load(emb_path)).float()
        else:
            emb_path_0 = os.path.join(self.emb_dir, f"{index}_0.npy")
            emb = torch.from_numpy(np.load(emb_path_0)).float()

        return img, emb


def get_uvit_t2i_dataloaders(
    config: DictConfig, model_cfg: DictConfig, dataset_cfg: DictConfig
) -> DataLoader:
    image_size = getattr(dataset_cfg, "image_size", 256)

    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = COCOWithEmbeddings(dataset_cfg, transform)

    g = torch.Generator()
    g.manual_seed(getattr(model_cfg, "seed", 0))

    return DataLoader(
        dataset,
        num_workers=config.dataloader_num_workers,
        batch_size=model_cfg.batch_size,
        shuffle=True,
        generator=g,
    )
