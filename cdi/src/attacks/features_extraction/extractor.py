import torch
from torch import Tensor as T
from typing import Tuple
from tqdm import tqdm
from src.attacks import DataSource
from src.models import DiffusionModel
from src.dataloaders import loaders


class FeatureExtractor(DataSource):
    """
    Base feature extractor class.
    Handles loading, iteration, and saving of extracted features.
    Supports datasets that return (image, label) or (image, embedding).
    """

    model: DiffusionModel

    def process_batch(self, batch: Tuple[T, T], *args, **kwargs) -> T:
        """
        Should be implemented by subclasses.
        Each subclass receives (images, targets) as a tuple.
        """
        raise NotImplementedError

    # ----------------------------------------------------------------------
    @staticmethod
    def normalize_batch(batch):
        """
        Ensure the batch returned by a DataLoader is always a 2-tuple: (images, targets).
        Supports (images, labels), (images, embeddings), or longer tuples.
        """
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 2:
                return batch[0], batch[1]
            elif len(batch) == 1:
                return batch[0], None
        return batch, None
    # ----------------------------------------------------------------------

    def process_data(self, *args, **kwargs) -> T:
        """
        Main feature extraction loop. Iterates over a DataLoader and aggregates features.
        """
        assert self.model is not None

        # Load DataLoader(s)
        loaders_tuple = loaders[self.model_cfg.dataloader](
            self.config, self.model_cfg, self.dataset_cfg
        )

        # All dataloaders now return a single DataLoader based on dataset_cfg.split
        # Legacy support: handle tuple returns for backwards compatibility
        if isinstance(loaders_tuple, (list, tuple)):
            split = getattr(self.dataset_cfg, "split", "train")
            if split == "val":
                loader = loaders_tuple[1]
                print(f"[INFO] Using validation loader ({split})")
            else:
                loader = loaders_tuple[0]
                print(f"[INFO] Using training loader ({split})")
        else:
            loader = loaders_tuple
            split = getattr(self.dataset_cfg, "split", "train")
            print(f"[INFO] Using {split} split")

        features = []
        samples_processed = 0
        total_batches = int(self.total_samples / self.model_cfg.batch_size + 1)

        # Main loop
        for batch in tqdm(loader, total=total_batches):
            images, targets = self.normalize_batch(batch)

            # Safety: ensure tensors
            if not torch.is_tensor(images):
                raise TypeError(
                    f"[ERROR] Expected tensor for images, got {type(images)} instead."
                )

            features.append(self.process_batch((images, targets), *args, **kwargs))
            samples_processed += images.shape[0]
            if samples_processed >= self.total_samples:
                break

        # Concatenate collected features
        return torch.cat(features, dim=0)[: self.total_samples]

    # ----------------------------------------------------------------------
    def check_data(self, data: T) -> None:
        """Check that the feature tensor has the expected structure."""
        assert len(data.shape) >= 3, (
            f"Expected at least 3D feature tensor, got {data.shape}"
        )
        assert data.shape[0] == self.total_samples, (
            f"Expected {self.total_samples} samples, got {data.shape[0]}"
        )

    # ----------------------------------------------------------------------
    def run(self, *args, **kwargs) -> None:
        """
        Run the full extraction pipeline:
          1. Collect features from dataloader
          2. Verify structure
          3. Save output
        """
        print("[INFO] Starting feature extraction...")
        features = self.process_data(*args, **kwargs)
        print("[INFO] Checking data consistency...")
        self.check_data(features)
        print("[INFO] Saving extracted features...")
        self.save(features)
        print("[✅] Feature extraction complete.")
