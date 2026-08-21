from src.local_datasets.dataset import ImageFolderDataset, ImageNetV2
from src.local_datasets.image_caption import ImageCaptionRegistry

datasets = {
    "imagenet": ImageFolderDataset,
    "imagenet_real": ImageFolderDataset,
    "imagenet_tp": ImageFolderDataset,
    "imagenet_fp": ImageFolderDataset,
    "imagenetv2": ImageNetV2,
    "cc12m_nitrot": ImageCaptionRegistry,
}
