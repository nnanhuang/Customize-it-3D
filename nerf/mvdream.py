# don't support yet
from transformers import CLIPTextModel, CLIPTokenizer, logging, CLIPVisionModel, CLIPFeatureExtractor
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.transforms as T
import math
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
import time
import os
import clip

from MVDream.mvdream.ldm.util import instantiate_from_config
from MVDream.mvdream.camera_utils import convert_opengl_to_blender, normalize_camera
from MVDream.mvdream.model_zoo import build_model

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

# load model
def load_model_from_config(config, ckpt, device, vram_O=False, verbose=False):

    pl_sd = torch.load(ckpt, map_location='cpu')

    if 'global_step' in pl_sd and verbose:
        print(f'[INFO] Global Step: {pl_sd["global_step"]}')

    sd = pl_sd['state_dict']

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print('[INFO] missing keys: \n', m)
    if len(u) > 0 and verbose:
        print('[INFO] unexpected keys: \n', u)

    # manually load ema and delete it to save GPU memory
    if model.use_ema:
        if verbose:
            print('[INFO] loading EMA...')
        model.model_ema.copy_to(model.model)
        del model.model_ema

    if vram_O:
        # TODO:we don't need decoder
        del model.first_stage_model.decoder

    torch.cuda.empty_cache()

    model.eval().to(device)

    return model

class MVDream(nn.Module):
    # n_view: int = 4
    # image_size: int = 256
    # guidance_scale: float = 50.0 

    def __init__(self, device, fp16,
                 config='./MVDream/mvdream/configs/sd-v2-base.yaml',
                 ckpt=None, vram_O=False, t_range=[0.02, 0.98], opt=None):
        super().__init__()

        self.device = device
        self.fp16 = fp16
        self.vram_O = vram_O
        self.t_range = t_range
        self.opt = opt
        self.camera_condition_type = "rotation"
        self.n_view = 4
        self.recon_std_rescale: float = 0.5
        self.config = config
        self.ckpt = ckpt
        self.mv_nums = 0
        
        print(f'[INFO] loading MVDream...')
        model_name = "sd-v2.1-base-4view"
        self.model = build_model(model_name, ckpt_path=None)
        
        for p in self.model.parameters():
            p.requires_grad_(False)

        # timesteps: use diffuser for convenience... hope it's alright.
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
    
        self.to(self.device)
        
        print("[INFO] Loaded MVDream Model!")

    def train_step(self, text_embeddings, pred_rgb, c2w, guidance_scale=50, as_latent=False, grad_scale=1, save_guidance_path:Path=None):
        # [B,4,4]
        camera = c2w 
        # adjust batch size
        text_embeddings = text_embeddings.repeat_interleave(4, dim=0)

        if as_latent:
            latents = F.interpolate(pred_rgb, (32, 32), mode='bilinear', align_corners=False) * 2 - 1
        else:
            # [B,3,256,256]-> [B,4,32,32]
            pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_256)

        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        t_in = t.repeat(text_embeddings.shape[0])

        with torch.no_grad():
            noise = torch.randn_like(latents)
            # [B,4,32,32],[B,4,32,32],[1]
            latents_noisy = self.model.q_sample(latents, t, noise)

            x_in = torch.cat([latents_noisy] * 2)
            
            # save input tensors for UNet
            if camera is not None:
                camera = self.get_camera_cond(camera)
                camera = camera.repeat(2,1).to(text_embeddings)
                context = {"context": text_embeddings, "camera": camera, "num_frames": self.n_view}
            else:
                context = {"context": text_embeddings}
            # text_embeddings:[2*B,77,1024],camera:[16,16]; num_frames = 4
            # x_in:[2*B,4,32,32]; t_in:[2*B]
            noise_pred = self.model.apply_model(x_in, t_in, context)
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        if not self.opt.no_recon_loss:
            # reconstruct x0
            latents_recon = self.model.predict_start_from_noise(latents_noisy, t, noise_pred)

            # clip or rescale x0
            if self.recon_std_rescale > 0:
                latents_recon_nocfg = self.model.predict_start_from_noise(latents_noisy, t, noise_pred_cond)
                latents_recon_nocfg_reshape = latents_recon_nocfg.view(-1,self.n_view, *latents_recon_nocfg.shape[1:])
                latents_recon_reshape = latents_recon.view(-1,self.n_view, *latents_recon.shape[1:])
                factor = (latents_recon_nocfg_reshape.std([1,2,3,4],keepdim=True) + 1e-8) / (latents_recon_reshape.std([1,2,3,4],keepdim=True) + 1e-8)
                
                latents_recon_adjust = latents_recon.clone() * factor.squeeze(1).repeat_interleave(self.n_view, dim=0)
                latents_recon = self.recon_std_rescale * latents_recon_adjust + (1-self.recon_std_rescale) * latents_recon

            # x0-reconstruction loss from Sec 3.2 and Appendix
            loss = 0.5 * F.mse_loss(latents, latents_recon.detach(), reduction="sum") / latents.shape[0]
            grad = torch.autograd.grad(loss, latents, retain_graph=True)[0] * grad_scale
        
        else:
            # Original SDS
            # w(t), sigma_t^2
            w = (1 - self.model.alphas_cumprod[t])
            grad = (grad_scale * w)[:, None, None, None] * (noise_pred - noise)
            grad = torch.nan_to_num(grad)

            latents.backward(gradient=grad, retain_graph=True)
            loss = grad.abs().mean().detach()
            self.mv_nums += 4
            
        return loss

    def angle_between(self, sph_v1, sph_v2):
        def sph2cart(sv):
            r, theta, phi = sv[0], sv[1], sv[2]
            return torch.tensor([r * torch.sin(theta) * torch.cos(phi), r * torch.sin(theta) * torch.sin(phi), r * torch.cos(theta)])
        def unit_vector(v):
            return v / torch.linalg.norm(v)
        def angle_between_2_sph(sv1, sv2):
            v1, v2 = sph2cart(sv1), sph2cart(sv2)
            v1_u, v2_u = unit_vector(v1), unit_vector(v2)
            return torch.arccos(torch.clip(torch.dot(v1_u, v2_u), -1.0, 1.0))
        sph_v2 = sph_v2.unsqueeze(0)
        angles = torch.empty(len(sph_v1), len(sph_v2)) # [1,3] [1,3]
        for i, sv1 in enumerate(sph_v1):
            for j, sv2 in enumerate(sph_v2):
                angles[i][j] = angle_between_2_sph(sv1, sv2)
        return angles

    # def decode_latents(self, latents):
    #     # zs: [B, 4, 32, 32] Latent space image
    #     # with self.model.ema_scope():
    #     imgs = self.model.decode_first_stage(latents)
    #     imgs = (imgs / 2 + 0.5).clamp(0, 1)

    #     return imgs # [B, 3, 256, 256] RGB space image

    def encode_imgs(self, imgs):
        # imgs: [B, 3, 256, 256] RGB space image
        # with self.model.ema_scope():
        imgs = imgs * 2 - 1
        latents = torch.cat([self.model.get_first_stage_encoding(self.model.encode_first_stage(img.unsqueeze(0))) for img in imgs], dim=0)
        return latents # [B, 4, 32, 32] Latent space image

    def get_camera_cond(self, 
            camera, #Float[Tensor, "B 4 4"]
    ):
        # Note: the input of threestudio is already blender coordinate system
        # camera = convert_opengl_to_blender(camera)
        if self.camera_condition_type == "rotation": # normalized camera
            camera = normalize_camera(camera)
            camera = camera.flatten(start_dim=1)
        else:
            raise NotImplementedError(f"Unknown camera_condition_type={self.camera_condition_type}")
        return camera
    