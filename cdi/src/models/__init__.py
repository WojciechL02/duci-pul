from typing import Dict

from src.models.GeneralLatentDiffusionWrapper import (
    GeneralLatentDiffusionWrapper as DiffusionModel,
)
from src.models.GeneralLatentDiffusionWrapper import GeneralDMProfiler as Profiler
from src.models.DiT_RFWrapper import DiT_RFWrapper as DiT_RFModel
from src.models.UViT_t2i_Wrapper import UViT_t2i_Wrapper as UViT_t2i_Model


diffusion_models: Dict[str, DiffusionModel] = {
    "dit_rf": DiT_RFModel,
    "dit_rf_g": DiT_RFModel,
    "uvit_t2i": UViT_t2i_Model,
    "uvit_t2i_deep": UViT_t2i_Model,
}
