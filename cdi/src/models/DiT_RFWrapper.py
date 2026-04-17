import os
from typing import Optional

import torch
from torch import Tensor as T
from torchvision.datasets.utils import download_url

from src.models import DiffusionModel, Profiler

from DiT_MoE.diffusion.rectified_flow import RectifiedFlow
from DiT_MoE.download import find_model
from DiT_MoE.models import (
    DiT_models,
    FLASH_ATTN_AVAILABLE,
    FLASH_ATTN_IMPORT_ERROR,
)
from diffusers.models import AutoencoderKL

class DiT_RFWrapper(DiffusionModel):
    """
    TODO: check vae training details - ema or mse
    """

    def __init__(self, model_cfg):
        super(DiT_RFWrapper, self).__init__(model_cfg)
        self.image_size = self.model_cfg.image_size
        self.latent_size = int(self.image_size) // 8
        self.arch = getattr(self.model_cfg, "arch", "DiT-XL/2")
        self.diffusion_mode = getattr(self.model_cfg, "diffusion_mode", "rf")
        self.num_classes = getattr(self.model_cfg, "num_classes", 1000)
        self.null_class = getattr(self.model_cfg, "null_class", self.num_classes)
        self.num_experts = getattr(self.model_cfg, "num_experts", 8)
        self.num_experts_per_tok = getattr(self.model_cfg, "num_experts_per_tok", 2)
        self.pretraining_tp = getattr(self.model_cfg, "pretraining_tp", 2)
        configured_flash_attn = getattr(self.model_cfg, "use_flash_attn", True)
        if not configured_flash_attn:
            raise ValueError(
                "DiT runs in CDI require flash attention; disabling it is unsupported."
            )
        if not FLASH_ATTN_AVAILABLE:
            raise ImportError(
                "flash_attn is required for DiT runs in CDI."
            ) from FLASH_ATTN_IMPORT_ERROR
        self.use_flash_attn = True
        self.model_dtype = self._resolve_dtype(
            getattr(self.model_cfg, "model_dtype", "float16")
        )
        self.vae_dtype = torch.float16
        self.base_dir = self._get_base_dir()
        self.cache_dir = self._resolve_path("./model_checkpoints/dit_rf")

        ckpt_path = self._ensure_checkpoint()
        self.model = DiT_models[self.arch](
            input_size=self.latent_size,
            num_classes=self.num_classes,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            pretraining_tp=self.pretraining_tp,
            use_flash_attn=self.use_flash_attn,
        )
        self.model = self.model.to(dtype=self.model_dtype)
        self._load_checkpoint(ckpt_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.vae = AutoencoderKL.from_pretrained(
            self.model_cfg.vae_model,
            cache_dir=self.cache_dir,
        ).to(self.device)
        self.vae = self.vae.to(dtype=self.vae_dtype)
        self.vae.eval()

        if self.diffusion_mode != "rf":
            raise ValueError(
                f"Unsupported diffusion mode: {self.diffusion_mode} (only 'rf' is supported)"
            )
        self.diffusion = RectifiedFlow(self.model)
        self.diffusion = self.diffusion.to(self.device)
        if self.model_dtype == torch.float16:
            self.diffusion = self.diffusion.half()

        if self.model_dtype in (torch.float16, torch.bfloat16):
            self._configure_half_precision_hooks()

        total_params = sum(p.numel() for p in self.model.parameters())
        print("TOTAL PARAMS: DiT MoE", total_params)

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

    def _resolve_dtype(self, dtype_name: str) -> torch.dtype:
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        try:
            return dtype_map[dtype_name.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported model dtype: {dtype_name}") from exc

    def _ensure_checkpoint(self) -> str:
        ckpt_path = self._resolve_path(self.model_cfg.diffuser_model)
        if os.path.isfile(ckpt_path):
            return ckpt_path

        checkpoint_url = getattr(self.model_cfg, "checkpoint_url", None)
        if checkpoint_url is None:
            raise FileNotFoundError(
                f"Checkpoint not found at {ckpt_path} and no checkpoint_url configured."
            )

        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        print(f"Checkpoint missing, downloading from {checkpoint_url}")
        download_url(
            checkpoint_url,
            os.path.dirname(ckpt_path),
            filename=os.path.basename(ckpt_path),
        )
        return ckpt_path

    def _remap_checkpoint_keys(self, state_dict):
        remapped_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            new_key = new_key.replace(".attn.qkv.", ".attn.Wqkv.")
            new_key = new_key.replace(".attn.proj.", ".attn.out_proj.")
            remapped_state_dict[new_key] = value
        return remapped_state_dict

    def _load_checkpoint(self, ckpt_path: str) -> None:
        print(f"Loading checkpoint: {ckpt_path}")
        state_dict = find_model(ckpt_path)
        remapped_state_dict = self._remap_checkpoint_keys(state_dict)
        missing, unexpected = self.model.load_state_dict(
            remapped_state_dict, strict=False
        )
        if missing:
            print(
                f"[Warning] Missing keys ({len(missing)}): "
                f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
            )
        if unexpected:
            print(
                f"[Warning] Unexpected keys ({len(unexpected)}): "
                f"{unexpected[:10]}{'...' if len(unexpected) > 10 else ''}"
            )
        print("Checkpoint loaded successfully with key remapping.")

    def _configure_half_precision_hooks(self) -> None:
        target_dtype = self.model_dtype

        def _cast_hook(_module, _inputs, output):
            return output.to(dtype=target_dtype)

        for name in ["t_embedder", "y_embedder"]:
            module = getattr(self.model, name, None)
            if module is not None:
                module.float()
                module.register_forward_hook(_cast_hook)

        x_embedder = getattr(self.model, "x_embedder", None)
        if x_embedder is not None:
            x_embedder.to(dtype=target_dtype)

    def _prepare_classes(self, classes: Optional[T], batch_size: int) -> T:
        if classes is None:
            return torch.full(
                (batch_size,),
                self.null_class,
                device=self.device,
                dtype=torch.long,
            )

        classes = classes.to(self.device)
        if classes.dim() == 1:
            classes = classes.long()
        return classes

    def _encode(self, images: T) -> T:
        latents = self.vae.encode(images.to(dtype=self.vae_dtype)).latent_dist.sample()
        return latents.mul_(0.18215).to(dtype=self.model_dtype)

    def _decode(self, latents: T) -> T:
        return self.vae.decode(latents.to(dtype=self.vae_dtype) / 0.18215).sample

    def noise_latents(self, latents: T, timestep: int, noise: T) -> T:
        latents = latents.to(self.device, dtype=self.model_dtype)
        noise = noise.to(self.device, dtype=self.model_dtype)

        x = latents
        b = x.size(0)
        t = torch.ones(latents.shape[0], device=self.device).long() * timestep / 1000
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        z1 = noise
        zt = (1 - texp) * x + texp * z1

        zt, t = zt.to(x.dtype), t.to(x.dtype)
        return zt

    def _predict_noise_from_latent(
        self, latents_noisy: T, classes: T, timestep: int
    ) -> T:
        classes = self._prepare_classes(classes, latents_noisy.shape[0])
        zt = latents_noisy.to(self.device, dtype=self.model_dtype)

        t = (
            torch.ones(latents_noisy.shape[0], device=self.device).long() * timestep
        ) / 1000
        t = t.to(zt.dtype)
        vtheta = self.diffusion.model(zt, t, classes)
        if self.diffusion.learn_sigma is True:
            vtheta, _ = vtheta.chunk(2, dim=1)
        return vtheta

    def get_loss(self, latents: T, classes: T, timestep: int, noise: T):
        latents = latents.to(self.device, dtype=self.model_dtype)
        noise = noise.to(self.device, dtype=self.model_dtype)

        x = latents
        b = x.size(0)
        t = torch.ones(latents.shape[0], device=self.device).long() * timestep / 1000
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        z1 = noise
        zt = (1 - texp) * x + texp * z1

        t = torch.ones(latents.shape[0], device=self.device).long() * timestep / 1000
        cond = self._prepare_classes(classes, latents.shape[0])

        zt = latents
        zt = zt.to(dtype=self.model_dtype)
        t = t.to(zt.dtype)
        vtheta = self.diffusion.model(zt, t, cond)
        if self.diffusion.learn_sigma is True:
            vtheta, _ = vtheta.chunk(2, dim=1)
        zt, t = zt.to(x.dtype), t.to(x.dtype)
        batchwise_mse = ((z1 - x - vtheta) ** 2).mean(
            dim=list(range(1, len(x.shape)))
        )
        return batchwise_mse.detach()


class DiTProfiler(Profiler):
    pass
