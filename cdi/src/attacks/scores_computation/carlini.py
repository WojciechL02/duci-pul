# https://arxiv.org/abs/2301.13188

from src.attacks import ScoreComputer
from torch import Tensor as T

from typing import Tuple


class CarliniLossThresholdComputer(ScoreComputer):
    def process_data(self, data: T) -> T:
        """
        Compute scores
        Inputs of shape (N_samples, N_measurements, 1)
        """
        return data.mean(dim=1).squeeze(1)
