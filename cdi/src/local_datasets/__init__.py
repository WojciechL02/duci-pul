from src.local_datasets.dataset import ImageFolderDataset, ImageNetV2

datasets = {
    "imagenet": ImageFolderDataset,
    "imagenet_real": ImageFolderDataset,
    "imagenet_tp": ImageFolderDataset,
    "imagenet_fp": ImageFolderDataset,
    "imagenetv2": ImageNetV2,
}
