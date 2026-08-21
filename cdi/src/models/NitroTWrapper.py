"""
CDI wrapper for AMD Nitro-T 1.2B MMDiT (amd/Nitro-T-1.2B) — a text-to-image
flow-matching model trained on a ~35M-image mixture containing CC12M, JourneyDB-train,
DiffusionDB, SA-1B, TextCaps. Native resolution is 1024x1024; we run at 512 to fit
GPU memory and to match the filter threshold we used to split members / non-members.

Components:
  * VAE:     diffusers.AutoencoderDC (DC-AE 32x, 32-channel latent, scale 0.41407)
  * Text:    transformers.LlamaForCausalLM (Llama 3.2 1B, 2048-dim hidden states)
  * DiT:     NitroMMDiTModel (SD3-style MMDiT, JointTransformerBlock, 1.26B params)

The CLiD attack uses the flow-matching velocity the DiT was trained to predict, with
the linear path  zt = (1-t_norm)*x + t_norm*noise,  t_norm = timestep / 1000.
For CFG the unconditional prompt is "" tokenized once and cached.
"""
from __future__ import annotations

import importlib.util
import os
from typing import Optional

import torch
from torch import Tensor as T

from src.models import DiffusionModel, Profiler


class NitroTWrapper(DiffusionModel):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)

        self.image_size = int(self.model_cfg.image_size)
        self.vae_downsample = int(getattr(self.model_cfg, "vae_downsample", 32))
        self.latent_size = self.image_size // self.vae_downsample
        self.model_dtype = self._resolve_dtype(getattr(self.model_cfg, "model_dtype", "bfloat16"))
        self.vae_dtype = self._resolve_dtype(getattr(self.model_cfg, "vae_dtype", "bfloat16"))
        self.text_dtype = self._resolve_dtype(getattr(self.model_cfg, "text_dtype", "bfloat16"))
        self.max_seq_length = int(getattr(self.model_cfg, "max_seq_length", 256))

        self.base_dir = self._get_base_dir()
        self.nitrot_dir = self._resolve_path(
            getattr(self.model_cfg, "nitrot_dir", "./model_checkpoints/nitrot/Nitro-T-1.2B")
        )
        self.llama_dir = self._resolve_path(
            getattr(self.model_cfg, "llama_dir", "./model_checkpoints/nitrot/Llama-3.2-1B")
        )

        # ---- VAE (DC-AE 32x) ----
        from diffusers import AutoencoderDC
        self.vae = AutoencoderDC.from_pretrained(
            os.path.join(self.nitrot_dir, "vae"), torch_dtype=self.vae_dtype
        ).to(self.device).eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae_scale_factor = float(self.vae.config.scaling_factor)  # 0.41407

        # ---- Text encoder (Llama 3.2 1B) ----
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.llama_dir)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.text_encoder = AutoModelForCausalLM.from_pretrained(
            self.llama_dir, torch_dtype=self.text_dtype
        ).to(self.device).eval()
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        # ---- Transformer (Nitro MMDiT, loaded from repo's custom `transformer.py`) ----
        tpath = os.path.join(self.nitrot_dir, "transformer", "transformer.py")
        spec = importlib.util.spec_from_file_location("_nitrot_tx", tpath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[arg-type]
        self.model = mod.NitroMMDiTModel.from_pretrained(
            os.path.join(self.nitrot_dir, "transformer"), torch_dtype=self.model_dtype
        ).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # ---- Cache empty-prompt embedding (CFG-dropout token in training) ----
        with torch.no_grad():
            self._null_embed = self._encode_texts([""])  # (1, L, 2048)

        total_dit = sum(p.numel() for p in self.model.parameters())
        total_vae = sum(p.numel() for p in self.vae.parameters())
        total_txt = sum(p.numel() for p in self.text_encoder.parameters())
        print(f"[NitroTWrapper] DiT={total_dit/1e6:.1f}M  VAE={total_vae/1e6:.1f}M  "
              f"TextEnc={total_txt/1e6:.1f}M  dtype={self.model_dtype}")

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
            "float16": torch.float16, "fp16": torch.float16,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float32": torch.float32, "fp32": torch.float32,
        }
        try:
            return table[name.lower()]
        except KeyError as e:
            raise ValueError(f"Unsupported dtype: {name}") from e

    def _encode_texts(self, prompts: list[str]) -> T:
        """Tokenize and run Llama to get last-hidden-states (B, L, 2048)."""
        tok = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.max_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        ids = tok.input_ids.to(self.device)
        with torch.no_grad():
            out = self.text_encoder(ids, output_hidden_states=True)
        return out.hidden_states[-1].to(dtype=self.text_dtype)

    # ------------------------------------------------------------------ #
    #                              CDI API                               #
    # ------------------------------------------------------------------ #

    def _encode(self, images: T) -> T:
        x = images.to(self.device, dtype=self.vae_dtype)
        with torch.no_grad():
            z = self.vae.encode(x).latent
        # Nitro-T's pipeline scales post-encode by vae.scaling_factor (see pipeline.py's
        # decode line: `image = vae.decode(latents / self.vae.scaling_factor).sample`).
        return (z * self.vae_scale_factor).to(dtype=self.model_dtype)

    def _decode(self, latents: T) -> T:
        z = latents.to(self.device, dtype=self.vae_dtype) / self.vae_scale_factor
        with torch.no_grad():
            return self.vae.decode(z).sample.to(dtype=torch.float32)

    def noise_latents(self, latents: T, timestep: int, noise: T) -> T:
        """Linear flow path (same as DiT-RF / minFM / DeltaFM):  zt = (1-t)*x + t*eps."""
        x = latents.to(self.device, dtype=self.model_dtype)
        n = noise.to(self.device, dtype=self.model_dtype)
        b = x.size(0)
        t = torch.full((b,), float(timestep) / 1000.0, device=self.device, dtype=x.dtype)
        t_exp = t.view(b, *([1] * (x.ndim - 1)))
        return (1 - t_exp) * x + t_exp * n

    def _prepare_classes(self, classes, batch_size) -> T:
        """`classes` is either a list of caption strings (conditional) or None (uncond)."""
        if classes is None:
            return self._null_embed.expand(batch_size, -1, -1).contiguous()
        if isinstance(classes, (list, tuple)):
            return self._encode_texts(list(classes))
        if isinstance(classes, torch.Tensor) and classes.dim() > 1 and classes.dtype.is_floating_point:
            # already precomputed embeddings
            return classes.to(self.device, dtype=self.text_dtype)
        # Fallback: treat as empty-prompt
        return self._null_embed.expand(batch_size, -1, -1).contiguous()

    def _predict_noise_from_latent(self, latents_noisy, classes, timestep):
        b = latents_noisy.shape[0]
        zt = latents_noisy.to(self.device, dtype=self.model_dtype)
        text_embeds = self._prepare_classes(classes, b).to(dtype=self.model_dtype)
        t = torch.full((b,), float(timestep), device=self.device, dtype=torch.long)
        # Note: we deliberately do NOT wrap this call in torch.no_grad. The base class
        # `predict_noise_from_latent` already controls gradient flow via `use_grad`,
        # and gradient_masking needs gradients to flow back to the noisy-latent input.
        out = self.model(
            hidden_states=zt,
            encoder_hidden_states=text_embeds,
            timestep=t,
            return_dict=True,
        )
        v_pred = out.sample
        if v_pred.dtype == torch.bfloat16:
            v_pred = v_pred.to(torch.float16)
        return v_pred

    def get_loss(self, latents: T, classes, timestep: int, noise: T) -> T:
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


class NitroTProfiler(Profiler):
    pass
