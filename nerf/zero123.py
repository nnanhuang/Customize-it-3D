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

from ldm.util import instantiate_from_config

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

class Zero123(nn.Module):
    def __init__(self, device, fp16,
                 config='./pretrained/zero123/sd-objaverse-finetune-c_concat-256.yaml',
                 ckpt='./pretrained/zero123/zero123-xl.ckpt', vram_O=False, t_range=[0.02, 0.98], opt=None):
        super().__init__()

        self.device = device
        self.fp16 = fp16
        self.vram_O = vram_O
        self.t_range = t_range
        self.opt = opt

        print(f'[INFO] loading zero123 using checkpoint {ckpt} ...')
        self.config = OmegaConf.load(config)
        self.model = load_model_from_config(self.config, ckpt, device=self.device, vram_O=vram_O)

        # timesteps: use diffuser for convenience... hope it's alright.
        self.num_train_timesteps = self.config.model.params.timesteps

        self.scheduler = DDIMScheduler(
            self.num_train_timesteps,
            self.config.model.params.linear_start,
            self.config.model.params.linear_end,
            beta_schedule='scaled_linear',
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        
        print(f'[INFO] loaded zero123!')

    def train_step(self, embeddings, pred_rgb, polar, azimuth, radius, guidance_scale=3, as_latent=False, grad_scale=1, save_guidance_path:Path=None):
        # pred_rgb: tensor [1, 3, H, W] in [0, 1]

        # adjust SDS scale based on how far the novel view is from the known view
        # embeddings = self.embeddings['zero123']['default']
        ref_radii = embeddings['ref_radi']
        ref_polars = embeddings['ref_polar']
        ref_azimuths = embeddings['ref_azimuth']
        v1 = torch.stack([radius + ref_radii, torch.deg2rad(polar + ref_polars), torch.deg2rad(azimuth + ref_azimuths)], dim=-1)   # polar,azimuth,radius are all actually delta wrt default
        v2 = torch.stack([torch.tensor(ref_radii), torch.deg2rad(torch.tensor(ref_polars)), torch.deg2rad(torch.tensor(ref_azimuths))], dim=-1)
        angles = torch.rad2deg(self.angle_between(v1, v2)).to(self.device)
        if self.opt.zero123_grad_scale == 'angle':
            grad_scale = (angles.min(dim=1)[0] / (180/1)) * grad_scale  # rethink 180/len(ref_azimuths) # claforte: try inverting grad_scale or just fixing it to 1.0
            # grad_scale = (angles.min(dim=1)[0] / (180/len(ref_azimuths))) * grad_scale  # rethink 180/len(ref_azimuths) # claforte: try inverting grad_scale or just fixing it to 1.0
        elif self.opt.zero123_grad_scale == 'None':
            grad_scale = 1.0 # claforte: I think this might converge faster...?
        elif self.opt.zero123_grad_scale == 'fix':
            grad_scale = grad_scale
        else:
            assert False, f'Unrecognized `zero123_grad_scale`: {self.opt.zero123_grad_scale}'
        
        if as_latent:
            latents = F.interpolate(pred_rgb, (32, 32), mode='bilinear', align_corners=False) * 2 - 1
        else:
            pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_256)

        t = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device)

        # # Set weights acc to closeness in angle
        # if len(ref_azimuths) > 1:
        #     inv_angles = 1/angles
        #     inv_angles[inv_angles > 100] = 100
        #     inv_angles /= inv_angles.max(dim=-1, keepdim=True)[0]
        #     inv_angles[inv_angles < 0.1] = 0
        # else:
        inv_angles = torch.tensor([1.]).to(self.device)

        # Multiply closeness-weight by user-given weights
        zero123_ws = torch.tensor(embeddings['zero123_ws']).to(self.device) * inv_angles
        # zero123_ws = torch.tensor(embeddings['zero123_ws'])[None, :].to(self.device) * inv_angles
        zero123_ws /= zero123_ws.max(dim=-1, keepdim=True)[0]
        zero123_ws[zero123_ws < 0.1] = 0

        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            x_in = torch.cat([latents_noisy] * 2)
            t_in = torch.cat([t] * 2)
            
            # for (zero123_w, c_crossattn, c_concat, ref_polar, ref_azimuth, ref_radius) in zip(zero123_ws.T, embeddings['c_crossattn'], embeddings['c_concat'], ref_polars, ref_azimuths, ref_radii):
            zero123_w = zero123_ws.T
            c_crossattn = embeddings['c_crossattn']
            c_concat = embeddings['c_concat']
            # polar,azimuth,radius are all actually delta wrt default
            p = polar 
            a = azimuth
            a[a > 180] -= 360 # range in [-180, 180]
            r = radius 
            # T = torch.tensor([math.radians(p), math.sin(math.radians(-a)), math.cos(math.radians(a)), r])
            # T = T[None, None, :].to(self.device)
            # T = torch.stack([torch.deg2rad(p), torch.sin(torch.deg2rad(-a)), torch.cos(torch.deg2rad(a)), r], dim=-1)[:, None, :]
            T = torch.stack([torch.deg2rad(p), torch.sin(torch.deg2rad(a)), torch.cos(torch.deg2rad(a)), r], dim=-1)[:, None, :]
            cond = {}
            clip_emb = self.model.cc_projection(torch.cat([c_crossattn.repeat(len(T), 1, 1), T], dim=-1))
            cond['c_crossattn'] = [torch.cat([torch.zeros_like(clip_emb).to(self.device), clip_emb], dim=0)]
            cond['c_concat'] = [torch.cat([torch.zeros_like(c_concat).repeat(len(T), 1, 1, 1).to(self.device), c_concat.repeat(len(T), 1, 1, 1)], dim=0)]
            noise_pred = self.model.apply_model(x_in, t_in, cond)
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            noise_pred = zero123_w[:, None, None, None] * noise_pred

        # noise_pred = torch.stack(noise_pred).sum(dim=0) / zero123_ws.sum(dim=-1)[:, None, None, None]

        w = (1 - self.alphas[t])
        grad = (grad_scale * w)[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        # import kiui
        # if not as_latent:
        #     kiui.vis.plot_image(pred_rgb_256)
        # kiui.vis.plot_matrix(latents)
        # kiui.vis.plot_matrix(grad)

        # import kiui
        # latents = torch.randn((1, 4, 32, 32), device=self.device)
        # kiui.lo(latents)
        # self.scheduler.set_timesteps(30)
        # with torch.no_grad():
        #     for i, t in enumerate(self.scheduler.timesteps):
        #         x_in = torch.cat([latents] * 2)
        #         t_in = torch.cat([t.view(1)] * 2).to(self.device)

        #         noise_pred = self.model.apply_model(x_in, t_in, cond)
        #         noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        #         noise_pred = noise_pred_uncond + 3 * (noise_pred_cond - noise_pred_uncond)

        #         latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        # imgs = self.decode_latents(latents)
        # print(polar, azimuth, radius)
        # kiui.vis.plot_image(pred_rgb_256, imgs)

        # if save_guidance_path:
        #     with torch.no_grad():
        #         if as_latent:
        #             pred_rgb_256 = self.decode_latents(latents) # claforte: test!

        #         # visualize predicted denoised image
        #         result_hopefully_less_noisy_image = self.decode_latents(self.model.predict_start_from_noise(latents_noisy, t, noise_pred))

        #         # visualize noisier image
        #         result_noisier_image = self.decode_latents(latents_noisy)

        #         # all 3 input images are [1, 3, H, W], e.g. [1, 3, 512, 512]
        #         viz_images = torch.cat([pred_rgb_256, result_noisier_image, result_hopefully_less_noisy_image],dim=-1)
        #         save_image(viz_images, save_guidance_path)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        # loss = SpecifyGradient.apply(latents, grad)
        latents.backward(gradient=grad, retain_graph=True)
        loss = grad.abs().mean().detach()

        return loss

    @torch.no_grad()
    def get_img_embeds(self, x):
        # x: image tensor [B, 3, 256, 256] in [0, 1]
        x = x * 2 - 1
        c_crossattn = self.model.get_learned_conditioning(x) #.tile(n_samples, 1, 1)
        c_concat = self.model.encode_first_stage(x).mode()
        return c_crossattn, c_concat

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

    def decode_latents(self, latents):
        # zs: [B, 4, 32, 32] Latent space image
        # with self.model.ema_scope():
        imgs = self.model.decode_first_stage(latents)
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs # [B, 3, 256, 256] RGB space image

    def encode_imgs(self, imgs):
        # imgs: [B, 3, 256, 256] RGB space image
        # with self.model.ema_scope():
        imgs = imgs * 2 - 1
        latents = torch.cat([self.model.get_first_stage_encoding(self.model.encode_first_stage(img.unsqueeze(0))) for img in imgs], dim=0)
        return latents # [B, 4, 32, 32] Latent space image
