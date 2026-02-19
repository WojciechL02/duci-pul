import torch
from torch import Tensor as T
from src.models import DiffusionModel, Profiler

from DiT_MoE.diffusion.rectified_flow import RectifiedFlow 
from diffusers.models import AutoencoderKL
from DiT_MoE.models import DiT_XL_2

from typing import Optional

class DiT_RFWrapper(DiffusionModel):
    """
    TODO: check vae training details - ema or mse
    """

    def __init__(self, model_cfg):
        super(DiT_RFWrapper, self).__init__(model_cfg)
        self.image_size = self.model_cfg.image_size
        self.latent_size = int(self.image_size) // 8

        self.vae = AutoencoderKL.from_pretrained(
            self.model_cfg.vae_model, cache_dir="./model_checkpoints/dit_rf"
        ).to(self.device).half()

        self.model = DiT_XL_2(input_size=self.latent_size, use_flash_attn=False).to(self.device).half()

        # === Dynamic checkpoint key remapping ===
        ckpt_path = self.model_cfg.diffuser_model
        print(f"Loading checkpoint: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        remapped_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            new_k = new_k.replace(".attn.Wqkv.", ".attn.qkv.")
            new_k = new_k.replace(".attn.out_proj.", ".attn.proj.")
            new_k = new_k.replace(".attn.q_norm.", ".attn.q_norm_layer.")
            new_k = new_k.replace(".attn.k_norm.", ".attn.k_norm_layer.")
            if "attn.Wqkv" in new_k:
                new_k = new_k.replace("attn.Wqkv", "attn.qkv")
            if "attn.out_proj" in new_k:
                new_k = new_k.replace("attn.out_proj", "attn.proj")
            remapped_state_dict[new_k] = v
        missing, unexpected = self.model.load_state_dict(remapped_state_dict, strict=False)
        if missing:
            print(f"[Warning] Missing keys ({len(missing)}): {missing[:10]}{'...' if len(missing)>10 else ''}")
        if unexpected:
            print(f"[Warning] Unexpected keys ({len(unexpected)}): {unexpected[:10]}{'...' if len(unexpected)>10 else ''}")
        print("Checkpoint loaded successfully with key remapping.")

        self.model.eval()
        self.diffusion = RectifiedFlow(self.model).half()

        def _half_hook(_m, _inp, out):
            return out.half()

        for name in ["t_embedder", "y_embedder"]:
            m = getattr(self.diffusion.model, name, None)
            if m is not None:
                m.float()
                m.register_forward_hook(_half_hook)

        x_emb = getattr(self.diffusion.model, "x_embedder", None)
        if x_emb is not None:
            x_emb.half()

        total_params = sum(p.numel() for p in self.model.parameters())
        print("TOTAL PARAMS: DiT MoE", total_params)

    def _encode(self, images: T) -> T:
        return self.vae.encode(images.half()).latent_dist.sample().mul_(0.18215)

    def _decode(self, latents: T) -> T:
        return self.vae.decode(latents.half() / 0.18215).sample

    def noise_latents(self, latents: T, timestep: int, noise: T) -> T:
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
        t = torch.ones(latents_noisy.shape[0], device=self.device).long() * timestep / 1000

        classes = (
            classes
            if classes is not None
            else torch.ones(latents_noisy.shape[0], device=self.device).long() * 1000
        )

        zt = latents_noisy
        cond = classes
        zt = zt.to(torch.float16)
        t = t.to(zt.dtype)
        vtheta = self.diffusion.model(zt, t, cond) 
        if self.diffusion.learn_sigma == True: 
            vtheta, _ = vtheta.chunk(2, dim=1) 
        return vtheta
    
    def predict_noise_from_latent(
        self,
        latents: T,
        classes: T,
        timestep: int,
        noise: Optional[T] = None,
        use_grad: bool = False,
    ) -> T:
        if noise is not None:
            latents_noisy = self.noise_latents(latents, timestep, noise)
        else:
            latents_noisy = latents

        if use_grad:
            noise_pred = self._predict_noise_from_latent(
                latents_noisy, classes, timestep
            )
        else:
            with torch.no_grad():
                noise_pred = self._predict_noise_from_latent(
                    latents_noisy, classes, timestep
                )
        return noise_pred 
    
    def get_loss(self, latents: T, classes: T, timestep: int, noise: T):
        x = latents
        b = x.size(0)
        t = torch.ones(latents.shape[0], device=self.device).long() * timestep / 1000
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        z1 = noise
        zt = (1 - texp) * x + texp * z1

        t = torch.ones(latents.shape[0], device=self.device).long() * timestep / 1000
        classes = (
            classes
            if classes is not None
            else torch.ones(latents.shape[0], device=self.device).long() * 1000
        )

        zt = latents
        cond = classes
        zt = zt.to(torch.float16)
        t = t.to(zt.dtype)
        vtheta = self.diffusion.model(zt, t, cond) 
        if self.diffusion.learn_sigma == True: 
            vtheta, _ = vtheta.chunk(2, dim=1) 
        zt, t = zt.to(x.dtype), t.to(x.dtype)
        batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        return batchwise_mse.detach()


class DiTProfiler(Profiler):
    pass
