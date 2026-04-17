import torch
from torch import Tensor as T
from src.models import DiffusionModel

import libs.autoencoder
import libs.clip
from libs.uvit_t2i import UViT as UViT_t2i
import numpy as np
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
from typing import Optional, Sequence


def stable_diffusion_beta_schedule(
    linear_start=0.00085, linear_end=0.0120, n_timestep=1000
):
    """
    stable_diffusion_beta_schedule from https://github.com/baofff/U-ViT/blob/main/train_ldm_discrete.py#L23
    """
    _betas = (
        torch.linspace(
            linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64
        )
        ** 2
    )
    return _betas.numpy()


def stp(s, ts: torch.Tensor):  # scalar tensor product
    """
    Scalar tensor product from https://github.com/baofff/U-ViT/blob/main/train_ldm_discrete.py#L42
    """
    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).type_as(ts)
    extra_dims = (1,) * (ts.dim() - 1)
    return s.view(-1, *extra_dims) * ts


def get_skip(alphas, betas):
    """
    From https://github.com/baofff/U-ViT/blob/main/train_ldm_discrete.py#L30
    """
    N = len(betas) - 1
    skip_alphas = np.ones([N + 1, N + 1], dtype=betas.dtype)
    for s in range(N + 1):
        skip_alphas[s, s + 1 :] = alphas[s + 1 :].cumprod()
    skip_betas = np.zeros([N + 1, N + 1], dtype=betas.dtype)
    for t in range(N + 1):
        prod = betas[1 : t + 1] * skip_alphas[1 : t + 1, t]
        skip_betas[:t, t] = (prod[::-1].cumsum())[::-1]
    return skip_alphas, skip_betas


class Schedule(object):  # discrete time
    """
    Discrite (normal - timestep is an int) schedule from https://github.com/baofff/U-ViT/blob/main/train_ldm_discrete.py#L53
    """

    def __init__(self, _betas):
        r"""_betas[0...999] = betas[1...1000]
        for n>=1, betas[n] is the variance of q(xn|xn-1)
        for n=0,  betas[0]=0
        """

        self._betas = _betas
        self.betas = np.append(0.0, _betas)
        self.alphas = 1.0 - self.betas
        self.N = len(_betas)

        assert isinstance(self.betas, np.ndarray) and self.betas[0] == 0
        assert isinstance(self.alphas, np.ndarray) and self.alphas[0] == 1
        assert len(self.betas) == len(self.alphas)

        self.skip_alphas, self.skip_betas = get_skip(self.alphas, self.betas)
        self.cum_alphas = self.skip_alphas[0]  # cum_alphas = alphas.cumprod()
        self.cum_betas = self.skip_betas[0]
        self.snr = self.cum_alphas / self.cum_betas

    def tilde_beta(self, s, t):
        return self.skip_betas[s, t] * self.cum_betas[s] / self.cum_betas[t]

    def sample(
        self, latents: T, timestep: np.int8, noise: T
    ) -> T:  # sample from q(xn|x0), where n is uniform
        latents_noisy = stp(self.cum_alphas[timestep] ** 0.5, latents) + stp(
            self.cum_betas[timestep] ** 0.5, noise
        )
        return latents_noisy

    def __repr__(self):
        return f"Schedule({self.betas[:10]}..., {self.N})"


class UViT_t2i_Wrapper(DiffusionModel):
    """
    TODO: check vae training details - ema or mse
    """

    def __init__(self, model_cfg):
        super(UViT_t2i_Wrapper, self).__init__(model_cfg)

        self.image_size = self.model_cfg.image_size
        self.latent_size = self.image_size // 8
        self.patch_size = 2 if self.image_size == 256 else 4

        self.uvit = UViT_t2i(
            img_size=self.latent_size,
            patch_size=self.patch_size,
            in_chans=4,
            embed_dim=512,
            depth=self.model_cfg.depth,
            num_heads=8,
            mlp_ratio=4,
            qkv_bias=False,
            mlp_time_embed=False,
            clip_dim=768,
            num_clip_token=77,
        )

        self.uvit.to(self.device)
        self.uvit.load_state_dict(
            torch.load(model_cfg.diffuser_model, map_location="cpu")
        )
        self.uvit.eval()

        total_params = sum(p.numel() for p in self.uvit.parameters())
        print("TOTAL PARAMS: uvit t2i", total_params)

        self.autoencoder = libs.autoencoder.get_model(model_cfg.vae_model)
        self.autoencoder.scale_factor = 0.23010
        self.autoencoder.to(self.device)
        self.autoencoder.eval()

        self._betas = stable_diffusion_beta_schedule()
        self.schedule = Schedule(self._betas)
        self._clip_embedder = None
        self._empty_context = None

    def _encode(self, images: T) -> T:
        """
        Encodes image batch into latents using first stage VAE model
        """
        return self.autoencoder.encode(images)

    def _decode(self, latents: T) -> T:
        """
        Decodes latents batch into images using first stage VAE model
        """
        return self.autoencoder.decode(latents)

    def noise_latents(self, latents: T, timestep: int, noise: T) -> T:
        return self.schedule.sample(
            latents, np.ones((latents.shape[0]), dtype=np.int64) * timestep, noise
        )

    def _predict_noise_from_latent(
        self, latents_noisy: T, classes: T, timestep: int
    ) -> T:
        """
        Predics noise for a noised latent at timestep t
        """
        timestep = torch.tensor(
            np.ones((latents_noisy.shape[0]), dtype=np.int64) * timestep,
            device=self.device,
        )
        noise_pred = self.uvit(latents_noisy, timestep, context=classes)
        return noise_pred

    def get_alpha_cumprod(self, t: int) -> float:
        """
        Return cumulative product of alphas for timestep t
        """
        return self.schedule.cum_alphas[t]

    def get_loss(self, latents: T, classes: T, timestep: int, noise: T) -> T:
        """
        Standard diffusion noise-prediction MSE for (latents, text_embeddings).
        """
        latents_noisy = self.noise_latents(latents, timestep, noise)
        noise_pred = self._predict_noise_from_latent(latents_noisy, classes, timestep)
        batchwise_mse = ((noise - noise_pred) ** 2).mean(dim=list(range(1, len(noise.shape))))
        return batchwise_mse.detach()

    def _get_clip_embedder(self):
        if self._clip_embedder is None:
            self._clip_embedder = libs.clip.FrozenCLIPEmbedder(device=self.device)
            self._clip_embedder.eval()
            self._clip_embedder.to(self.device)
        return self._clip_embedder

    @torch.no_grad()
    def encode_prompts(self, prompts: Sequence[str]) -> T:
        if isinstance(prompts, str):
            prompts = [prompts]
        return self._get_clip_embedder().encode(list(prompts))

    @torch.no_grad()
    def get_empty_context(
        self, batch_size: int, dtype: Optional[torch.dtype] = None
    ) -> T:
        if self._empty_context is None:
            self._empty_context = self.encode_prompts([""])[0]
        empty_context = self._empty_context.unsqueeze(0).repeat(batch_size, 1, 1)
        if dtype is not None:
            empty_context = empty_context.to(dtype=dtype)
        return empty_context

    @torch.no_grad()
    def sample_from_contexts(
        self,
        contexts: T,
        sample_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        seeds: Optional[Sequence[int]] = None,
    ) -> T:
        contexts = contexts.to(self.device)
        if seed is not None and seeds is not None:
            raise ValueError("Pass either `seed` or `seeds`, not both.")

        # The supplementary SD-MIA pipeline seeds each generated sample separately.
        if seeds is not None:
            if len(seeds) != contexts.size(0):
                raise ValueError(
                    f"Expected {contexts.size(0)} seeds, got {len(seeds)}."
                )
            z_init = torch.stack(
                [
                    torch.randn(
                        4,
                        self.latent_size,
                        self.latent_size,
                        device=self.device,
                        generator=torch.Generator(device=self.device).manual_seed(
                            int(item_seed)
                        ),
                    )
                    for item_seed in seeds
                ],
                dim=0,
            )
        else:
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed)

            z_init = torch.randn(
                contexts.size(0),
                4,
                self.latent_size,
                self.latent_size,
                device=self.device,
                generator=generator,
            )
        noise_schedule = NoiseScheduleVP(
            schedule="discrete",
            betas=torch.tensor(self._betas, device=self.device).float(),
        )
        empty_context = self.get_empty_context(
            contexts.size(0), dtype=contexts.dtype
        ).to(self.device)

        def model_fn(x, t_continuous):
            t = t_continuous * self.schedule.N
            cond = self.uvit(x, t, context=contexts)
            if guidance_scale == 0:
                return cond
            uncond = self.uvit(x, t, context=empty_context)
            return cond + guidance_scale * (cond - uncond)

        dpm_solver = DPM_Solver(
            model_fn, noise_schedule, predict_x0=True, thresholding=False
        )
        latents = dpm_solver.sample(
            z_init, steps=sample_steps, eps=1.0 / self.schedule.N, T=1.0
        )
        images = self.decode(latents).float()
        return (0.5 * (images + 1.0)).clamp_(0.0, 1.0)

    @torch.no_grad()
    def sample_from_prompts(
        self,
        prompts: Sequence[str],
        sample_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        seeds: Optional[Sequence[int]] = None,
    ) -> T:
        contexts = self.encode_prompts(prompts)
        return self.sample_from_contexts(
            contexts,
            sample_steps=sample_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            seeds=seeds,
        )
