"""
CDI wrapper for DeltaFM (Contrastive Flow Matching, ICCV 2025) on ImageNet-1k.

Backbone: REPA SiT-XL/2 (675M params) + contrastive-FM loss with a DINOv2 projector
that is only used at training time. Official checkpoint:
  gstoica3/DeltaFM/checkpoints/repa_sitxl2_deltafm_256.pt  (state dict under key "ema")
VAE: stabilityai/sd-vae-ft-mse (4-channel latent, 0.18215 scale, same as DiT_RFWrapper).

Flow path is linear, same convention as SiT/REPA and DiT_RFWrapper:
    zt = (1 - t) * x_clean + t * noise,  t in [0, 1] == timestep / 1000.
SiT.forward returns (v_pred, zs, labels); CLiD only needs v_pred.
"""
from __future__ import annotations

import os
import sys

import torch
from torch import Tensor as T

from src.models import DiffusionModel, Profiler

_CDI_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DELTAFM_DIR = os.path.join(_CDI_ROOT, "vendored", "DeltaFM")


def _load_deltafm_sit_models():
    """Load `SiT_models` from the vendored DeltaFM repo without clashing with minFM's
    top-level `models` package (MinFMWrapper adds `cdi/vendored/minFM` to sys.path)."""
    import importlib.util
    sit_path = os.path.join(_DELTAFM_DIR, "models", "sit.py")
    if not os.path.exists(sit_path):
        raise FileNotFoundError(f"vendored DeltaFM sit.py missing at {sit_path}")
    spec = importlib.util.spec_from_file_location("_deltafm_sit", sit_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module.SiT_models


SiT_models = _load_deltafm_sit_models()


class DeltaFMWrapper(DiffusionModel):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)

        self.image_size = int(self.model_cfg.image_size)
        self.latent_size = self.image_size // 8

        self.arch = getattr(self.model_cfg, "arch", "SiT-XL/2")
        self.num_classes = int(getattr(self.model_cfg, "num_classes", 1000))
        self.null_class = int(getattr(self.model_cfg, "null_class", self.num_classes))
        self.encoder_depth = int(getattr(self.model_cfg, "encoder_depth", 8))
        self.projector_dims = [
            int(d)
            for d in str(getattr(self.model_cfg, "projector_embed_dims", "768")).split(",")
        ]

        self.model_dtype = self._resolve_dtype(getattr(self.model_cfg, "model_dtype", "float32"))
        self.vae_dtype = self._resolve_dtype(getattr(self.model_cfg, "vae_dtype", "float16"))

        self.base_dir = self._get_base_dir()
        self.ckpt_path = self._resolve_path(
            getattr(
                self.model_cfg,
                "diffuser_model",
                "./model_checkpoints/deltafm/checkpoints/repa_sitxl2_deltafm_256.pt",
            )
        )

        self.model = SiT_models[self.arch](
            input_size=self.latent_size,
            num_classes=self.num_classes,
            use_cfg=True,
            z_dims=self.projector_dims,
            encoder_depth=self.encoder_depth,
            fused_attn=bool(getattr(self.model_cfg, "fused_attn", True)),
            qk_norm=bool(getattr(self.model_cfg, "qk_norm", False)),
        )
        self._load_checkpoint(self.model, self.ckpt_path)
        self.model = self.model.to(self.device, dtype=self.model_dtype).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        from diffusers.models import AutoencoderKL

        vae_id = getattr(self.model_cfg, "vae_model", "stabilityai/sd-vae-ft-mse")
        vae_cache = self._resolve_path(
            getattr(self.model_cfg, "vae_cache", "./model_checkpoints/deltafm/vae_cache")
        )
        os.makedirs(vae_cache, exist_ok=True)
        self.vae = (
            AutoencoderKL.from_pretrained(vae_id, cache_dir=vae_cache)
            .to(self.device, dtype=self.vae_dtype)
            .eval()
        )
        for p in self.vae.parameters():
            p.requires_grad = False

        total = sum(p.numel() for p in self.model.parameters())
        print(f"[DeltaFMWrapper] SiT params: {total / 1e6:.2f}M, dtype={self.model_dtype}")

    def _get_base_dir(self) -> str:
        try:
            from hydra.utils import get_original_cwd

            return get_original_cwd()
        except Exception:
            return os.getcwd()

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self.base_dir, path))

    @staticmethod
    def _resolve_dtype(name: str) -> torch.dtype:
        table = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        try:
            return table[name.lower()]
        except KeyError as e:
            raise ValueError(f"Unsupported dtype: {name}") from e

    def _load_checkpoint(self, model, ckpt_path):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"DeltaFM checkpoint not found: {ckpt_path}")
        print(f"[DeltaFMWrapper] loading {ckpt_path}")
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(blob, dict) and "ema" in blob:
            sd = blob["ema"]
        elif isinstance(blob, dict) and "model" in blob:
            sd = blob["model"]
        else:
            sd = blob
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[DeltaFMWrapper] missing ({len(missing)}): {missing[:5]}")
        if unexpected:
            print(f"[DeltaFMWrapper] unexpected ({len(unexpected)}): {unexpected[:5]}")

    def _encode(self, images: T) -> T:
        x = images.to(self.device, dtype=self.vae_dtype)
        with torch.no_grad():
            latents = self.vae.encode(x).latent_dist.sample()
        return (latents * 0.18215).to(dtype=self.model_dtype)

    def _decode(self, latents: T) -> T:
        z = latents.to(self.device, dtype=self.vae_dtype) / 0.18215
        with torch.no_grad():
            return self.vae.decode(z).sample.to(dtype=torch.float32)

    def noise_latents(self, latents: T, timestep: int, noise: T) -> T:
        x = latents.to(self.device, dtype=self.model_dtype)
        n = noise.to(self.device, dtype=self.model_dtype)
        b = x.size(0)
        t = torch.full((b,), float(timestep) / 1000.0, device=self.device, dtype=x.dtype)
        t_exp = t.view(b, *([1] * (x.ndim - 1)))
        return (1 - t_exp) * x + t_exp * n

    def _prepare_classes(self, classes, batch_size):
        if classes is None:
            return torch.full(
                (batch_size,),
                self.null_class,
                device=self.device,
                dtype=torch.long,
            )
        cls = classes.to(self.device)
        if cls.dim() > 1:
            cls = cls.view(batch_size)
        return cls.long()

    def _predict_noise_from_latent(self, latents_noisy, classes, timestep):
        b = latents_noisy.shape[0]
        zt = latents_noisy.to(self.device, dtype=self.model_dtype)
        t = torch.full(
            (b,), float(timestep) / 1000.0, device=self.device, dtype=zt.dtype
        )
        y = self._prepare_classes(classes, b)
        v_pred, *_ = self.model(zt, t, y)
        return v_pred

    def get_loss(self, latents: T, classes, timestep: int, noise: T) -> T:
        """Per-sample flow-matching MSE used by carlini_lt and multiple_loss."""
        x = latents.to(self.device, dtype=self.model_dtype)
        n = noise.to(self.device, dtype=self.model_dtype)
        zt = self.noise_latents(x, timestep, n)
        with torch.no_grad():
            v_pred = self._predict_noise_from_latent(zt, classes, timestep)
        v_truth = (n - x).to(dtype=v_pred.dtype)
        mse = ((v_truth.float() - v_pred.float()) ** 2).mean(
            dim=list(range(1, x.ndim))
        )
        return mse.detach()


class DeltaFMProfiler(Profiler):
    pass
