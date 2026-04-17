from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose
import os
import torch

try:
    from imagenetv2_pytorch import ImageNetV2Dataset
except ImportError:
    ImageNetV2Dataset = None


class ImageFolderDataset:
    """
    Standard ImageNet dataset loader using torchvision ImageFolder.
    
    Expects directory structure:
        dataset_path/
            train/
                n01440764/
                    image1.JPEG
                    image2.JPEG
                    ...
                n01443537/
                    ...
            val/
                n01440764/
                    ...
    """
    def __init__(self, dataset_cfg, transform: Compose):
        self.dataset = ImageFolder(
            os.path.join(dataset_cfg.dataset_path, dataset_cfg.split),
            transform=transform,
        )
        self.collate_fn = None


class ImageNetV2:
    def __init__(self, dataset_cfg, transform: Compose):
        if ImageNetV2Dataset is None:
            raise ImportError("imagenetv2_pytorch is required for ImageNetV2 dataset")
        self.dataset_cfg = dataset_cfg
        self.dataset = ImageNetV2Dataset(
            variant=self.dataset_cfg.split,
            transform=transform,
            location=self.dataset_cfg.dataset_path,
        )
        self.collate_fn = None
