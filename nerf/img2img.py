import torch
from PIL import Image
import os

from diffusers import StableDiffusionImg2ImgPipeline,StableDiffusionXLImg2ImgPipeline

def img2img_sd(rgb_files,prompt,device,H,W,output_dir,trained_model_path):
    # model_id_or_path = "stabilityai/stable-diffusion-2-base"
    prompt = prompt + ", no face, rgb photo"
    model_id_or_path = trained_model_path
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    
    new_rgb_files = []
    for i in range(len(rgb_files)):
        image_pil = Image.open(rgb_files[i]).convert("RGB")
        image_pil = image_pil.resize((H, W))
        if i in [4, 5, 6, 15, 16, 17, 26, 27, 28]:           
            images = pipe(prompt=prompt, image=image_pil, strength=0.15, guidance_scale=7.5).images
            file_path = os.path.join(output_dir,os.path.basename(rgb_files[i]))
            images[0].save(file_path)
            new_rgb_files.append(file_path)
        else:
            images = pipe(prompt=prompt, image=image_pil, strength=0.05, guidance_scale=7.5).images
            file_path = os.path.join(output_dir,os.path.basename(rgb_files[i]))
            images[0].save(file_path)
            new_rgb_files.append(file_path)
        
    return new_rgb_files
    
def img2img_sdxl(rgb_files,prompt,device,H,W,output_dir):
    model_id_or_path = 'stabilityai/stable-diffusion-xl-base-1.0'
    # model_id_or_path = 'stabilityai/stable-diffusion-xl-refiner-1.0'
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_id_or_path, torch_dtype=torch.float16
    )
    pipe = pipe.to(device)
    
    for i in range(len(rgb_files)):
        image_pil = Image.open(rgb_files[i]).convert("RGB")
        image_pil = image_pil.resize((H, W))

        image = pipe(prompt, image=image_pil,strength=0.15,original_size=(H,W),target_size=(H,W)).images[0]
        
        image.save(os.path.join(output_dir,f'stone_dragon_statue_{i}.png'))

if __name__=='__main__':
    device = "cuda"
    H = 800
    W = 800
    prompt = "a beautiful and cute pink barbie doll"       
    rgb_files = ['results/doll/mvimg/doll_ep0070_0000_rgb.png',
                 'results/doll/mvimg/doll_ep0070_0001_rgb.png',
                 'results/doll/mvimg/doll_ep0070_0002_rgb.png',
                 'results/doll/mvimg/doll_ep0070_0003_rgb.png',
                 ]
    
    sd_output_dir = 'sd'
    # sdxl_output_dir = 'sdxl'
    os.makedirs(sd_output_dir,exist_ok=True)
    # os.makedirs(sdxl_output_dir,exist_ok=True)

    img2img_sd(rgb_files,prompt,device,H,W,sd_output_dir,'out/doll')
    # img2img_sdxl(rgb_files,prompt,device,H,W,sdxl_output_dir)

    