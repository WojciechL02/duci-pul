# https://arxiv.org/abs/2301.13188

from src.attacks import FeatureExtractor
from torch import Tensor as T
import torch
from typing import Tuple


class CarliniLossThresholdExtractor(FeatureExtractor):
    def process_batch(self, batch: Tuple[T, T]) -> T:
        """
        TODO: create docstring
        """
        # --- FIX: make unpacking robust ---
        images, classes = batch[:2]
        # -----------------------------------

        bs = images.shape[0]
        images = images.to(self.device)
        if isinstance(classes, T):
            classes = classes.to(self.device)

        latents = self.model.encode(images)

        losses = []
        for _ in range(self.attack_cfg.n_repetitions):
            noise = torch.randn_like(latents).to(self.device)
            loss = self.model.get_loss(
                latents, classes, self.attack_cfg.timestep, noise
            ).detach().cpu()
            losses.append(loss)

        losses = torch.stack(losses, dim=1).unsqueeze(2)  # B, N_measurements, 1 feature
        return losses
