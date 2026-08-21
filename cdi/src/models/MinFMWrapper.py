"""
CDI wrapper for minFM flux-tiny DiT (https://github.com/Kai-46/minFM) trained on ImageNet-1k.

Exposes the same API that `DiT_RFWrapper` does, so CLiD / Carlini-LT / gradient-masking
feature extractors can consume it unchanged. Flow matching uses the same "clean at t=0,
noise at t=1" convention as `DiT_RFWrapper`, so `noise_latents(zt = (1-t)*x + t*noise)` and
`_predict_noise_from_latent` (returning velocity) are structurally identical to DiT-RF.

Text conditioning: minFM's flux-tiny on ImageNet uses a lookup-table "text encoder"
(`VanillaEmbedder`) of precomputed CLIP embeddings keyed by ImageNet class label. The
same 768-d embedding serves as both the per-token `txt` stream and the global `vec`
stream of the Flux denoiser. For CFG / CLiD unconditional passes the wrapper uses the
special index 1000 which stores the empty-string ("") CLIP embedding used at training
time (txt_drop_prob = 0.1).
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import torch
from einops import rearrange
from torch import Tensor as T

from src.models import DiffusionModel, Profiler

_CDI_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MINFM_DIR = os.path.join(_CDI_ROOT, "vendored", "minFM")
if os.path.isdir(_MINFM_DIR) and _MINFM_DIR not in sys.path:
    sys.path.insert(0, _MINFM_DIR)

# minFM internals (vendored; see cdi/vendored/minFM)
from models.flux_denoiser import FluxDenoiser, FluxDenoiserParams  # noqa: E402
from models.flux_vae import AutoEncoder, AutoEncoderParams  # noqa: E402
from models.patchifier import Patchifier, PatchifierParams  # noqa: E402


_NULL_PROMPT_INDEX = 1000  # VanillaEmbedder stores "" embedding at row 1000 (label_to_txt[1000] == "")


class MinFMWrapper(DiffusionModel):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)

        self.image_size = int(self.model_cfg.image_size)
        self.num_classes = int(getattr(self.model_cfg, "num_classes", 1000))
        self.null_class = int(getattr(self.model_cfg, "null_class", _NULL_PROMPT_INDEX))

        self.patch_size = tuple(getattr(self.model_cfg, "patch_size", [1, 2, 2]))
        self.vae_compression = tuple(getattr(self.model_cfg, "vae_compression_factors", [1, 8, 8]))
        self.vae_latent_channels = int(getattr(self.model_cfg, "vae_latent_channels", 16))
        self.vae_scale_factor = float(getattr(self.model_cfg, "vae_scale_factor", 0.3611))
        self.vae_shift_factor = float(getattr(self.model_cfg, "vae_shift_factor", 0.1159))

        self.model_dtype = self._resolve_dtype(getattr(self.model_cfg, "model_dtype", "bfloat16"))
        self.vae_dtype = self._resolve_dtype(getattr(self.model_cfg, "vae_dtype", "bfloat16"))

        self.base_dir = self._get_base_dir()
        self.ckpt_root = self._resolve_path(
            getattr(self.model_cfg, "ckpt_root", "./model_checkpoints/minfm")
        )

        # VAE (skip minFM's built-in safetensors loader — FLUX.1-dev is gated; we load from diffusers below)
        vae_params = AutoEncoderParams(
            in_channels=3,
            z_channels=self.vae_latent_channels,
            scale_factor=self.vae_scale_factor,
            shift_factor=self.vae_shift_factor,
        )
        self.vae = AutoEncoder(vae_params, from_pretrained=False)
        self._load_flux_vae_weights(self.vae)
        self.vae = self.vae.to(self.device, dtype=self.vae_dtype).eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        # Patchifier (no parameters, just metadata)
        self.patchifier = Patchifier(
            PatchifierParams(
                patch_size=self.patch_size,
                vae_latent_channels=self.vae_latent_channels,
                vae_compression_factors=self.vae_compression,
            )
        )

        # Text embedding lookup (VanillaEmbedder baked down to a raw nn.Embedding)
        self.text_embedding = self._load_text_embeddings().to(self.device, dtype=self.model_dtype)

        # Denoiser
        denoiser_params = FluxDenoiserParams(
            d_model=int(getattr(self.model_cfg, "d_model", 1024)),
            d_head=int(getattr(self.model_cfg, "d_head", 64)),
            n_ds_blocks=int(getattr(self.model_cfg, "n_ds_blocks", 8)),
            n_ss_blocks=int(getattr(self.model_cfg, "n_ss_blocks", 16)),
            d_txt=int(getattr(self.model_cfg, "d_txt", 768)),
            d_vec=int(getattr(self.model_cfg, "d_vec", 768)),
            d_img=int(getattr(self.model_cfg, "d_img", 64)),
            rope_axis_dim=list(getattr(self.model_cfg, "rope_axis_dim", [8, 28, 28])),
            guidance_embed=bool(getattr(self.model_cfg, "guidance_embed", False)),
        )
        self.model = FluxDenoiser(denoiser_params)
        self._load_denoiser_weights(self.model)
        self.model = self.model.to(self.device, dtype=self.model_dtype).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        total = sum(p.numel() for p in self.model.parameters())
        print(f"[MinFMWrapper] FluxDenoiser params: {total / 1e6:.2f}M, dtype={self.model_dtype}")

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

    def _load_text_embeddings(self) -> torch.Tensor:
        meta_path = self._resolve_path(
            getattr(self.model_cfg, "clip_embeddings_path", "./model_checkpoints/minfm/ilsvrc2012_clip_only.pt")
        )
        data = torch.load(meta_path, map_location="cpu", weights_only=False)
        emb: torch.Tensor = data["clip_embeddings"]
        assert emb.shape == (1001, 768), f"expected (1001, 768) CLIP table, got {tuple(emb.shape)}"
        return emb.contiguous()

    def _load_flux_vae_weights(self, vae: AutoEncoder) -> None:
        """Load FLUX VAE weights in BFL flat format (same format minFM itself uses)."""
        from safetensors.torch import load_file

        vae_path = self._resolve_path(
            getattr(self.model_cfg, "vae_ckpt", os.path.join(self.ckpt_root, "flux_vae_bfl", "ae.safetensors"))
        )
        if not os.path.exists(vae_path):
            raise FileNotFoundError(
                f"FLUX VAE weights not found at {vae_path}. "
                "Download ae.safetensors (BFL format) from e.g. ModelsLab/flux.1-dev on HF."
            )
        print(f"[MinFMWrapper] loading FLUX VAE weights from {vae_path}")
        sd = load_file(vae_path)
        missing, unexpected = vae.load_state_dict(sd, strict=False)
        if missing:
            raise RuntimeError(f"FLUX VAE missing keys (first 5): {missing[:5]}")
        if unexpected:
            print(f"[MinFMWrapper] FLUX VAE unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    def _load_denoiser_weights(self, model: FluxDenoiser) -> None:
        ckpt_name = getattr(self.model_cfg, "denoiser_ckpt", "flat/denoiser_ema.pt")
        path = self._resolve_path(os.path.join(self.ckpt_root, ckpt_name))
        print(f"[MinFMWrapper] loading denoiser ckpt: {path}")
        sd = torch.load(path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[MinFMWrapper] denoiser missing ({len(missing)}): {missing[:5]}")
        if unexpected:
            print(f"[MinFMWrapper] denoiser unexpected ({len(unexpected)}): {unexpected[:5]}")

    # ------------------------------------------------------------------ #
    #                              CDI API                               #
    # ------------------------------------------------------------------ #

    def _encode(self, images: T) -> T:
        """images (B, 3, H, W) in [-1, 1] -> latents (B, 16, H/8, W/8) in the model's dtype."""
        x = images.to(self.device, dtype=self.vae_dtype)
        with torch.no_grad():
            z = self.vae.encode(x)
        return z.to(dtype=self.model_dtype)

    def _decode(self, latents: T) -> T:
        z = latents.to(self.device, dtype=self.vae_dtype)
        with torch.no_grad():
            return self.vae.decode(z).to(dtype=torch.float32)

    def noise_latents(self, latents: T, timestep: int, noise: T) -> T:
        """Flow-matching forward: zt = (1 - t_norm) * x + t_norm * noise. t_norm = timestep / 1000."""
        x = latents.to(self.device, dtype=self.model_dtype)
        n = noise.to(self.device, dtype=self.model_dtype)
        b = x.size(0)
        t = torch.full((b,), float(timestep) / 1000.0, device=self.device, dtype=x.dtype)
        t_exp = t.view(b, *([1] * (x.ndim - 1)))
        return (1 - t_exp) * x + t_exp * n

    def _prepare_classes(self, classes: Optional[T], batch_size: int) -> T:
        if classes is None:
            return torch.full(
                (batch_size,), self.null_class, device=self.device, dtype=torch.long
            )
        cls = classes.to(self.device)
        if cls.dim() > 1:
            cls = cls.view(batch_size)
        return cls.long()

    def _patchify_batch(self, latents: T) -> tuple[T, T, T]:
        """(B, C, H, W) -> flat tokens + datum_lens + position_ids for FluxDenoiser."""
        b, c, h, w = latents.shape
        pf, ph, pw = self.patch_size
        gf, gh, gw = 1, h // ph, w // pw
        # Add trivial frame axis (f=1) to match Patchifier's expected 5D shape
        x5 = latents.unsqueeze(2)  # (B, C, 1, H, W)
        tokens = rearrange(
            x5,
            "b c (gf pf) (gh ph) (gw pw) -> (b gf gh gw) (c pf ph pw)",
            gf=gf, pf=pf, gh=gh, ph=ph, gw=gw, pw=pw,
        )
        per_datum = gf * gh * gw
        datum_lens = torch.full((b,), per_datum, device=latents.device, dtype=torch.int32)
        # Position IDs in (t, y, x) order: t=0 everywhere (single frame), y, x grid
        ys = torch.arange(gh, device=latents.device, dtype=torch.int32)
        xs = torch.arange(gw, device=latents.device, dtype=torch.int32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        tyx = torch.stack(
            [torch.zeros_like(yy), yy, xx], dim=-1
        ).reshape(-1, 3)  # (gh*gw, 3)
        position_ids = tyx.repeat(b, 1)  # (B * gh * gw, 3)
        return tokens, datum_lens, position_ids

    def _unpatchify_batch(
        self, tokens: T, b: int, h: int, w: int
    ) -> T:
        pf, ph, pw = self.patch_size
        gf, gh, gw = 1, h // ph, w // pw
        x5 = rearrange(
            tokens,
            "(b gf gh gw) (c pf ph pw) -> b c (gf pf) (gh ph) (gw pw)",
            b=b, gf=gf, pf=pf, gh=gh, ph=ph, gw=gw, pw=pw, c=self.vae_latent_channels,
        )
        return x5.squeeze(2)  # (B, C, H, W)

    def _predict_noise_from_latent(
        self, latents_noisy: T, classes: T, timestep: int
    ) -> T:
        """Run FluxDenoiser and return velocity predictions in the (B, C, H, W) latent shape.

        CLiD only uses the L2 distance between the prediction and the sampled noise; returning
        velocity (as trained) is statistically equivalent to returning (noise - x_clean),
        mirroring what `DiT_RFWrapper` does for the rectified-flow DiT checkpoints.
        """
        b, c, h, w = latents_noisy.shape
        zt = latents_noisy.to(self.device, dtype=self.model_dtype)

        cls_ids = self._prepare_classes(classes, b)  # (B,) long
        cls_emb = self.text_embedding[cls_ids]  # (B, 768)

        tokens, datum_lens, position_ids = self._patchify_batch(zt)
        tokens = tokens.to(dtype=self.model_dtype)

        txt = cls_emb  # (B, 768) == one-token-per-datum
        txt_datum_lens = torch.ones(b, device=self.device, dtype=torch.int32)
        txt_position_ids = torch.zeros(b, 3, device=self.device, dtype=torch.int32)

        t_norm = torch.full(
            (b,), float(timestep) / 1000.0, device=self.device, dtype=torch.float32
        )
        vec = cls_emb.to(dtype=self.model_dtype)

        v_pred_flat = self.model(
            txt=txt,
            txt_datum_lens=txt_datum_lens,
            txt_position_ids=txt_position_ids,
            img=tokens,
            img_datum_lens=datum_lens,
            img_position_ids=position_ids,
            t=t_norm,
            vec=vec,
            sequence_balancer=None,
        )

        # numpy doesn't support bfloat16, and CLiD stores L2 distances directly as numpy arrays.
        # Downstream only uses the L2 norms, so casting to fp16 here is safe and preserves range.
        v_pred = self._unpatchify_batch(v_pred_flat, b=b, h=h, w=w)
        if v_pred.dtype == torch.bfloat16:
            v_pred = v_pred.to(torch.float16)
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


class MinFMProfiler(Profiler):
    pass
