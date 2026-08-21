import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from PIL import Image
import numpy as np

from src.local_datasets import datasets
from src.local_datasets.dataset import ImageFolderDataset


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def get_dit_dataloader(
    config: DictConfig, model_cfg: DictConfig, dataset_cfg: DictConfig
) -> DataLoader:
    """
    Retrun dataloader matching DiT model
    """

    # Setup data
    transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda pil_image: center_crop_arr(pil_image, model_cfg.image_size)
            ),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),
        ]
    )

    # Create datasets
    dataset_cls = datasets.get(dataset_cfg.name, ImageFolderDataset)
    dataset = dataset_cls(dataset_cfg, transform)
    collate_fn = dataset.collate_fn
    dataset = dataset.dataset

    # Create data loaders for members and nonmembers
    g = torch.Generator()
    g.manual_seed(model_cfg.seed)
    return DataLoader(
        dataset,
        num_workers=config.dataloader_num_workers,
        batch_size=model_cfg.batch_size,
        shuffle=True,
        generator=g,
        collate_fn=collate_fn,
    )
