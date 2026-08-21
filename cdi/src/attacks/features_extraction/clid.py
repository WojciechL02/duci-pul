# https://arxiv.org/abs/2301.13188

from src.attacks import FeatureExtractor
from torch import Tensor as T
import torch
from typing import Tuple

from itertools import product


class CLiDExtractor(FeatureExtractor):
    def process_batch(self, batch: Tuple[T, T]) -> T:
        """
        TODO: create docstring
        """
        images, classes = batch[:2]
        B = images.shape[0]
        images = images.to(self.device)
        if type(classes) == T:
            classes = classes.to(self.device)

        latents = self.model.encode(images)

        L_xc, D_xcci = [], []
        for _, timestep in product(
            range(self.attack_cfg.n_repetitions), self.attack_cfg.timesteps
        ):
            noise = torch.randn_like(latents).to(self.device)
            noise_pred_cond = self.model.predict_noise_from_latent(
                latents, classes, timestep, noise
            )
            # Pick the unconditional form based on class encoding:
            #  - 1D int tensor (ImageNet-style class IDs) -> None, wrapper fills null_class.
            #  - higher-dim tensor (e.g., pre-tokenized prompts) -> zeros of same shape.
            #  - list of strings (T2I captions) -> None, wrapper uses empty-prompt embedding.
            if isinstance(classes, T):
                uncond_classes = None if classes.dim() == 1 else torch.zeros_like(classes)
            else:
                uncond_classes = None
            noise_pred_uncond = self.model.predict_noise_from_latent(
                latents, uncond_classes, timestep, noise
            )
            L_xc.append(
                -torch.norm(
                    noise_pred_cond.reshape(B, -1) - noise.reshape(B, -1), dim=1, p=2
                ).cpu()
            )
            D_xcci.append(
                torch.norm(
                    noise_pred_uncond.reshape(B, -1) - noise.reshape(B, -1), dim=1, p=2
                ).cpu()
                - torch.norm(
                    noise_pred_cond.reshape(B, -1) - noise.reshape(B, -1), dim=1, p=2
                ).cpu()
            )

        L_xc = torch.stack(L_xc, dim=1).mean(dim=1)
        D_xcci = torch.stack(D_xcci, dim=1).mean(dim=1)

        features = []
        for alpha in self.attack_cfg.alphas:
            features.append(alpha * D_xcci + (1 - alpha) * L_xc)

        features = torch.stack(features, dim=1).unsqueeze(1)  # B, 1, N_measurements
        return features
