"""Identity attack that records the absolute image path for each sample.

Produces .npz files with the same naming convention as feature files, but
``data`` is a 1-D object array of path strings instead of a numeric tensor.
This lets you look up exactly which image produced each feature value::

    paths = np.load("dit_rf_image_paths_5k_imagenet_5pc_train.npz",
                     allow_pickle=True)["data"]
    # paths[i] == absolute path of the image behind features[i]
"""

import os
import numpy as np
from tqdm import tqdm

from src.attacks.features_extraction.extractor import FeatureExtractor
from src.dataloaders import loaders


class ImagePathsExtractor(FeatureExtractor):

    def process_batch(self, batch, *args, **kwargs):
        raise NotImplementedError("ImagePathsExtractor does not process batches")

    def process_data(self, *args, **kwargs):
        loader_fn = loaders[self.model_cfg.dataloader]
        loaders_tuple = loader_fn(self.config, self.model_cfg, self.dataset_cfg)

        if isinstance(loaders_tuple, (list, tuple)):
            split = getattr(self.dataset_cfg, "split", "train")
            loader = loaders_tuple[1] if split == "val" else loaders_tuple[0]
        else:
            loader = loaders_tuple

        split = getattr(self.dataset_cfg, "split", "train")
        print(f"[INFO] Collecting image paths for {split} split")

        all_paths = []
        total_batches = int(self.total_samples / self.model_cfg.batch_size + 1)

        for batch in tqdm(loader, total=total_batches):
            normalized = self.normalize_batch(batch)
            if len(normalized) < 3:
                raise RuntimeError(
                    "Dataloader did not return paths as the third element. "
                    "Make sure the dataset's __getitem__ returns (image, target, path)."
                )
            paths = normalized[2]
            all_paths.extend(paths)

            if len(all_paths) >= self.total_samples:
                break

        return all_paths[: self.total_samples]

    def check_data(self, data) -> None:
        assert len(data) == self.total_samples, (
            f"Expected {self.total_samples} paths, got {len(data)}"
        )

    def save(self, data) -> None:
        path_array = np.array(data, dtype=object)

        os.makedirs(os.path.dirname(self.path_out) or ".", exist_ok=True)
        np.savez(self.path_out, data=path_array, metadata=self.metadata)
        print(f"[INFO] Saved {len(path_array)} image paths to {self.path_out}")
