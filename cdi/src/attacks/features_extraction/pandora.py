from src.attacks import FeatureExtractor
from torch import Tensor as T
import torch
from math import sqrt
from typing import Tuple, List
from torch.autograd import Variable


class PandoraExtractor(FeatureExtractor):
    def obtain_gradient_norms(self,latents: T, classes: T, noise: T, timestep: int):
        """
        Compute norms of gradients with respect x, theta
        Note: takes advantage of the fact that norm([a,b])=norm([norm(a),norm(b)])
            
        Returns:
            torch.Tensor or list: gradient norms of input ID
                [x_grad_norm_p1, ..., x_grad_norm_pN, theta_grad_norm_p1, ..., theta_grad_norm_pN, layer1_grad_norm_p1, ..., layerM_grad_norm_p1, ..., layer1_grad_norm_pN, ..., layerM_grad_norm_pN]
        """

        # Compute gradients with respect to latent input
        input_latents = (
                self.model.noise_latents(latents, timestep, noise)
                .clone()
                .detach()
                .requires_grad_(True)
            )
        
        self.model.zero_grad()
        noise_pred = self.model.predict_noise_from_latent(
                input_latents, classes, timestep, noise, use_grad=True
            ) 
        loss: T = torch.norm(noise_pred - noise, dim=(1, 2, 3), p=2).mean()
        loss.backward()       
        latents_grad = input_latents.grad.detach()

        # Compute gradients with respect to different layers
        self.model.zero_grad()
        noise_pred = self.model.predict_noise_from_latent(
                input_latents, classes, timestep, noise, use_grad=True
            ) 
        loss: T = torch.norm(noise_pred - noise, dim=(1, 2, 3), p=2).mean()
        loss.backward()         
        
        norms = [1,2,float("inf")]
        layer_norms = {p:[] for p in norms} # p: [layer_1_norm, layer_2_norm, ...]
        for i, (name,param) in enumerate(self.model.named_parameters()):

            if param.grad is None:
                pass
            else:
                grad = param.grad.flatten()
                        
                # Append all norms of layers to dictionary
                for p in norms:
                    layer_norms[p].append(torch.norm(grad,p=p))
        
        del noise_pred, input_latents, noise
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        for p in norms: # p: [x_grad_norm, theta_grad_norm, layer_1_grad_norm, ... layer_N_grad_norm]
            layer_norms[p] = [torch.norm(latents_grad,p=p)] + [torch.norm(torch.tensor(layer_norms[p]),p=p)] + layer_norms[p]
        
        # layer_norms = torch.tensor([l for p in norms for l in layer_norms[p]]) # [p1 first grad, ..., p1 last grad, ..., pM first grad, ... pM last grad]
        layer_norms = torch.tensor([layer_norms[p][l] for l in range(len(layer_norms[norms[0]])) for p in norms]) # [x_grad_norm_p1, ..., x_grad_norm_pM, theta_grad_norm_p1, ..., theta_grad_norm_pM, layer1_grad_norm_p1, ..., layer1_grad_norm_pM, ..., layerN_grad_norm_p1, ..., layerN_grad_norm_pM]
        return layer_norms


    def process_batch(self, batch: Tuple[T, T]) -> T:
        images, classes = batch[:2]

        assert images.shape[0]==1

        bs = images.shape[0]
        images = images.to(self.device)
        if type(classes) == T:
            classes = classes.to(self.device)
        latents = self.model.encode(images)

        noise = torch.randn_like(latents).to(self.device)

        timestep = self.attack_cfg.timestep
        gradient_norms = self.obtain_gradient_norms(latents, classes, noise, timestep)
        gradient_norms = gradient_norms.unsqueeze(0).unsqueeze(0)
        return gradient_norms
