# https://arxiv.org/pdf/2302.01316.pdf

from src.attacks import FeatureExtractor
from torch import Tensor as T
import torch
from typing import Tuple, List
from math import sqrt
import os
import numpy as np


class SecMIExtractor(FeatureExtractor):
    def single_step(
        self, latents: T, classes: T, timestep: int, target_timestep: int
    ) -> T:
        """
        TODO: Create a docstring
        """
        alpha_t = self.model.get_alpha_cumprod(timestep)
        alpha_target = self.model.get_alpha_cumprod(target_timestep)

        epsilon = self.model.predict_noise_from_latent(latents, classes, timestep)

        f_z_t = (latents - (sqrt(1 - alpha_t) * epsilon)) / sqrt(alpha_t)
        z_t_target = sqrt(alpha_target) * f_z_t + sqrt(1 - alpha_target) * epsilon

        return z_t_target  # B, C, H, W

    def multi_step(self, latents: T, classes: T, timesteps: List[int]) -> T:
        z_t_target = latents.clone().to(self.device)
        for timestep, target_timestep in zip(timesteps[:-1], timesteps[1:]):
            z_t_target = self.single_step(
                z_t_target, classes, timestep, target_timestep
            )

        return z_t_target  # B, C, H, W

    def process_batch(self, batch: Tuple[T, T]) -> T:
        images, classes = batch[:2]
        bs = images.shape[0]
        images = images.to(self.device)
        if type(classes) == T:
            classes = classes.to(self.device)
        latents = self.model.encode(images)

        target_timesteps = list(
            range(
                self.attack_cfg.t0,
                self.attack_cfg.t + self.attack_cfg.step,
                self.attack_cfg.step,
            )
        )
        z_det = self.multi_step(latents, classes, target_timesteps)
        z_step = self.single_step(
            z_det, classes, self.attack_cfg.t, self.attack_cfg.t + self.attack_cfg.step
        )
        z_step = self.single_step(
            z_step, classes, self.attack_cfg.t + self.attack_cfg.step, self.attack_cfg.t
        )

        return torch.cat(
            [z_det.detach().cpu().unsqueeze(1), z_step.detach().cpu().unsqueeze(1)],
            dim=1,
        )  # B, 2, C, H, W

    def process_data(self) -> Tuple[T, T]:
        second_attack = (
            "secmi_nn" if self.attack_cfg.name == "secmi_stat" else "secmi_stat"
        )
        path = os.path.join(
            *self.path_out.split("/")[:-1],
            f"{self.model_cfg.name}_{second_attack}_{self.config.run_id}_{self.dataset_cfg.name}_{self.dataset_cfg.split}.npz",
        )
        try:
            data = np.load(path, allow_pickle=True)
            print(f"Reading pre-computed features from {second_attack} attack, {path=}")
            return torch.from_numpy(data["data"])
        except FileNotFoundError:
            pass
        print(f"Computing features for {self.attack_cfg.name} attack")
        return super().process_data()
