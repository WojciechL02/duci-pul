from src.attacks import FeatureExtractor
from torch import Tensor as T
import torch
import os
import numpy as np
from typing import Tuple


class CombinationAttackExtractor(FeatureExtractor):
    def load_data(self, attack: str, folder: str) -> T:
        """
        Load the data from the file
        # TODO add Q_sus and Q_ctrl (if we still do it this way)
        """
        data = np.load(
            os.path.join(
                self._strip_hydra_path(folder),
                f"{self.model_cfg.name}_{attack}_{self.config.run_id}_{self.dataset_cfg.name}_{self.dataset_cfg.split}.npz",
            ),
            allow_pickle=True,
        )["data"]
        return torch.from_numpy(data)

    def process_data(self) -> Tuple[T, T]:
        features = []
        if self.attack_cfg.attacks_scores_to_use is not None:
            for attack in self.attack_cfg.attacks_scores_to_use.split(","):
                data = self.load_data(attack, self.config.path_to_scores)
                features.append(data.view(-1, 1, 1))

        if self.attack_cfg.attacks_features_to_use is not None:
            for attack in self.attack_cfg.attacks_features_to_use.split(","):
                data = self.load_data(attack, self.config.path_to_features)
                B = data.shape[0]
                features.append(data.view(B, 1, -1))

        return torch.cat(features, dim=2)  # B, 1, N_features
