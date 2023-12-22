import torch
import argparse

from nerf.provider import NeRFDataset
from nerf.utils import *
import torchvision.transforms as T
from scipy.ndimage import median_filter
# BLIP
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

# torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_pcd',action='store_true', help="using mesh sampled point cloud")
    parser.add_argument('--refine_W', type=int, default=800, help="GUI width")
    parser.add_argument('--refine_H', type=int, default=800, help="GUI width")
    parser.add_argument('--process_mask',action='store_true', help="process mask use SAM")
    parser.add_argument('--text', default=None, help="text prompt")
    parser.add_argument('--class_name', default=None, type=str, help="the class name used in dreambooth")
    parser.add_argument('--trained_model_path', default=None, type=str, help="the path of learned model")
    parser.add_argument('--edge_threshold', type=float, default=0.1,
                    help="remove edges with value > threshold")
    parser.add_argument('--edge_width', type=float, default=5,
                    help="edge width")
    parser.add_argument('--no_recon_loss',action='store_true', help="don't use recon_loss in mvdream guidance")
    parser.add_argument('--negative', default='', type=str, help="negative text prompt")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--final', action='store_true', help="final train mode")
    parser.add_argument('--refine', action='store_true', help="refine mode")
    parser.add_argument('--save_mesh', action='store_true', help="export an obj mesh with texture")
    parser.add_argument('--eval_interval', type=int, default=10, help="evaluate on the valid set every interval epochs")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--guidance', type=str,nargs='*', default=['stable-diffusion','zero123'], help='choose from [stable-diffusion, clip, zero123]')
    parser.add_argument('--zero123_config', type=str,
                    default='./pretrained/zero123/sd-objaverse-finetune-c_concat-256.yaml', help="config file for zero123")
    parser.add_argument('--zero123_ckpt', type=str,
                    default='./pretrained/zero123/zero123-xl.ckpt', help="ckpt for zero123")
    parser.add_argument('--zero123_grad_scale', type=str, default='angle',
                    help="whether to scale the gradients based on 'angle' or 'None' or 'fix' ")
    parser.add_argument('--mvdream_config', type=str,
                    default='./MVDream/mvdream/configs/sd-v2-base.yaml', help="config file for MVDream")
    parser.add_argument('--mvdream_ckpt', type=str,
                    default=None, help="ckpt for MVDream")

    parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--depth_model', type=str, default='dpt_hybrid', help='choose from [dpt_large, dpt_hybrid]')
    parser.add_argument('--guidance_scale', type=float,nargs='*', default=[10,5],help="diffusion model classifier-free guidance scale")
    parser.add_argument('--lambda_guidance', type=float, nargs='*',
                    default=[1,40], help="loss scale for SDS")
    parser.add_argument('--need_back', action='store_true', help="use back text prompt")
    parser.add_argument('--suppress_face', action='store_true', help="also use negative dir text prompt.")
    parser.add_argument('--ref_path', default=None, type=str, help="use image as referance, only support alpha image")


    ### training options
    parser.add_argument('--iters', type=int, default=10000, help="training iters")
    parser.add_argument('--refine_iters', type=int, default=3000, help="refine iters")
    parser.add_argument('--lr', type=float, default=1e-3, help="max learning rate")
    parser.add_argument('--min_lr', type=float, default=1e-4, help="minimal learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=512, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=64, help="num steps sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=32, help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
    parser.add_argument('--albedo_iters', type=int, default=1000, help="training iters that only use albedo shading")
    parser.add_argument('--uniform_sphere_rate', type=float, default=0.5, help="likelihood of sampling camera location uniformly on the sphere surface area")
    parser.add_argument('--diff_iters', type=int, default=400, help="training iters that only use albedo shading")
    parser.add_argument('--step_range', type=float, nargs='*', default=[0.2, 0.6])
    parser.add_argument('--normal_iters', type=int, default=0, help="training iters that only use normal shading")
    parser.add_argument('--batch_size', type=int, default=1, help="images to render per batch using NeRF")

    # model options
    parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--blob_density', type=float, default=5, help="max (center) density for the gaussian density blob")
    parser.add_argument('--blob_radius', type=float, default=0.2, help="control the radius for the gaussian density blob")
    # network backbone
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--backbone', type=str, default='tcnn', choices=['grid', 'tcnn', 'sdf', 'vanilla', 'normal'], help="nerf backbone")
    parser.add_argument('--optim', type=str, default='adan', choices=['adan', 'adam', 'adamw'], help="optimizer")
    parser.add_argument('--sd_version', type=str, default='2.0', choices=['1.5', '2.0'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    # rendering resolution in training, decrease this if CUDA OOM.
    parser.add_argument('--w', type=int, default=128, help="render width for NeRF in training")
    parser.add_argument('--h', type=int, default=128, help="render height for NeRF in training")
    
    ### dataset options
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--dt_gamma', type=float, default=0, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.1, help="minimum near distance for camera")
    parser.add_argument('--radius_range', type=float, nargs='*', default=[1.0, 1.5], help="training camera radius range")
    parser.add_argument('--fov', type=float, default=60, help="training camera fovy range")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[50, 70], help="training camera fovy range")
    parser.add_argument('--theta_range', type=float, nargs='*', default=[70, 110], help="training camera phi range")
    parser.add_argument('--phi_range', type=float, nargs='*', default=[-180, 180], help="training camera phi range")
    parser.add_argument('--default_radius', type=float, default=1,
                    help="radius for the default view") 
    parser.add_argument('--default_polar', type=float,
                        default=90, help="polar for the default view, 90 is front")
    parser.add_argument('--default_azimuth', type=float,
                        default=0, help="azimuth for the default view")
    
    parser.add_argument('--lambda_entropy', type=float, default=1, help="loss scale for alpha entropy")
    parser.add_argument('--lambda_opacity', type=float, default=1e-3, help="loss scale for alpha value")
    parser.add_argument('--lambda_orient', type=float, default=1e-2, help="loss scale for orientation")
    parser.add_argument('--lambda_smooth', type=float, default=1, help="loss scale for surface smoothness")
    parser.add_argument('--lambda_img', type=float, default=1e3, help="loss scale for ref loss")
    parser.add_argument('--lambda_depth', type=float, default=1, help="loss scale for depth loss")
    parser.add_argument('--lambda_clip', type=float, default=1, help="loss scale for clip loss")
    parser.add_argument('--lambda_normal_smooth2d', type=float, default=0,
                    help="loss scale for second-order 2D normal image smoothness")
    
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--radius', type=float, default=3, help="default GUI camera radius from center")
    parser.add_argument('--light_theta', type=float, default=60, help="default GUI light direction in [0, 180], corresponding to elevation [90, -90]")
    parser.add_argument('--light_phi', type=float, default=0, help="default GUI light direction in [0, 360), azimuth")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")
    parser.add_argument('--max_depth', type=float, default=10.0, help="farthest depth")
    
    opt = parser.parse_args()
    opt.cuda_ray = True
    optDict = opt.__dict__
    opt.experiment = opt.workspace
    opt.workspace = os.path.join('results', opt.workspace)
    if opt.workspace is not None:
        os.makedirs(opt.workspace, exist_ok=True) 
    
    # optimization for low VRAM usage
    opt.vram_O = False
    
    # opt.ref_paths, opt.ref_radii, opt.ref_polars, opt.ref_azimuths, opt.zero123_ws = [], [], [], [], []
    opt.default_zero123_w = 1
    opt.use_mvdream = False

    # parameters for image-conditioned generation
    if opt.ref_path is not None :
        if 'zero123' in opt.guidance:
            # fix fov as zero123 doesn't support changing fov
            opt.fovy_range = [opt.fov, opt.fov]
        # else:
        #     opt.known_view_interval = 2

        if 'stable-diffusion' in opt.guidance:
            opt.step_range = [0.2, 0.6]
            # don't use background model at sphere
            opt.bg_radius = -1

        if 'mvdream' in opt.guidance:
            opt.use_mvdream = True

        if opt.ref_path is not None:
            opt.ref_radii = opt.default_radius
            opt.ref_polars = opt.default_polar
            opt.ref_azimuths = opt.default_azimuth
            opt.zero123_ws = opt.default_zero123_w


    # reset to None
    if len(opt.ref_path) == 0:
        opt.ref_path = None
   
    if opt.backbone == 'vanilla':
        from nerf.network import NeRFNetwork
    elif opt.backbone == 'tcnn':
        from nerf.network_tcnn import NeRFNetwork
    else:
        raise NotImplementedError(f'--backbone {opt.backbone} is not implemented!')

    print(opt)
    seed_everything(opt.seed)

    # # load depth network
    # net_w = net_h = 384
    # depth_model = DPTDepthModel(
    #     path="dpt_weights/dpt_hybrid-midas-501f0c75.pt",
    #     backbone="vitb_rn50_384",
    #     non_negative=True,
    #     enable_attention_hooks=False,
    # )
    # depth_transform = T.Compose(
    # [
    #     T.Resize((384, 384)),
    #     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ]
    # )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # depth_model.to(device)

    if opt.optim == 'adan':
        from optimizer import Adan
        # Adan usually requires a larger LR
        optimizer = lambda model: Adan(model.get_params(5 * opt.lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
    else: # adam
        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

    if opt.backbone == 'vanilla':
        warm_up_with_cosine_lr = lambda iter: iter / opt.warm_iters if iter <= opt.warm_iters \
            else max(0.5 * ( math.cos((iter - opt.warm_iters) /(opt.iters - opt.warm_iters) * math.pi) + 1), 
                        opt.min_lr / opt.lr)

        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, warm_up_with_cosine_lr)
    else:
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1) # fixed
        # scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

    # init guidance
    guidance = nn.ModuleDict()
    lambda_guidance, guidance_scale = {}, {}
    for idx, guidance_type in enumerate(opt.guidance):
        lambda_guidance[guidance_type] = opt.lambda_guidance[idx] if idx < len(
            opt.lambda_guidance) else opt.lambda_guidance[-1]
        guidance_scale[guidance_type] = opt.guidance_scale[idx] if idx < len(
            opt.guidance_scale) else opt.guidance_scale[-1]
        
        if guidance_type == 'stable-diffusion':
            from nerf.sd import StableDiffusion
            guidance['stable-diffusion'] = StableDiffusion(opt.trained_model_path, opt.class_name, device, opt.sd_version, opt.hf_key, step_range=opt.step_range)
        elif guidance_type == 'zero123':
            from nerf.zero123 import Zero123
            guidance['zero123'] = Zero123(device=device, fp16=opt.fp16, config=opt.zero123_config,
                                            ckpt=opt.zero123_ckpt, vram_O=opt.vram_O, t_range=opt.step_range, opt=opt)
        elif guidance_type == 'mvdream':
            from nerf.mvdream import MVDream
            guidance['mvdream'] = MVDream(device,opt.fp16,opt.mvdream_config,vram_O=opt.vram_O, t_range=opt.step_range, opt=opt)
        elif guidance_type == 'clip':
            from nerf.clip import CLIP
            guidance['clip'] = CLIP(device)
        else:
            raise NotImplementedError(f'--guidance {opt.guidance} is not implemented.')

    # dict
    opt.lambda_guidance = lambda_guidance
    opt.guidance_scale = guidance_scale

    ref_imgs = cv2.imread(opt.ref_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
    image_pil = Image.open(opt.ref_path).convert("RGB")

    # generated caption
    if opt.text == None:
        print("load blip2 for image caption...")
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to("cuda")
        inputs = processor(image_pil, return_tensors="pt").to("cuda", torch.float16)
        out = blip_model.generate(**inputs)
        caption = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        caption = caption.replace("there is ", "")
        caption = caption.replace("close up", "photo")
        for d in ["black background", "white background"]:
            if d in caption:
                caption = caption.replace(d, "ground")
        print("Caption: ", caption)
        opt.text = caption

    with open(os.path.join(opt.workspace, 'setting.txt'), 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in optDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


    # only support alpha photo input.
    imgs = cv2.cvtColor(ref_imgs, cv2.COLOR_BGRA2RGBA)
    imgs = cv2.resize(imgs, (512, 512), interpolation=cv2.INTER_AREA)
    ref_imgs = (torch.from_numpy(imgs)/255.).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    # rgb
    ori_imgs = ref_imgs[:, :3, :, :] * ref_imgs[:, 3:, :, :] + (1 - ref_imgs[:, 3:, :, :])
    
    mask = imgs[:, :, 3:]
    # mask[mask < 0.5 * 255] = 0
    # mask[mask >= 0.5 * 255] = 1 
    kernel = np.ones(((5,5)), np.uint8) ##5
    mask = cv2.erode(mask,kernel,iterations=1)
    mask = (mask == 0)
    # cv2.imwrite('mask/depth_mask.png',mask*255)
    mask = (torch.from_numpy(mask)).unsqueeze(0).unsqueeze(0).to(device)
    depth_mask = mask
    
    midas_mask = imgs[..., 3] > 0.5
    midas_mask = torch.from_numpy(midas_mask).to(device)

    # load midas generated depth
    depth_path = opt.ref_path.replace('rgba', 'depth')
    if os.path.exists(depth_path):
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) # 800,800
        depth = cv2.resize(depth, (512, 512), interpolation=cv2.INTER_AREA) 
        depth = 1 - torch.from_numpy(depth.astype(np.float32) / 255).to(device)
        if len(depth.shape) == 4 and depth.shape[-1] > 1:
            depth = depth[..., 0]
            print(f'[WARN] dataset: {depth_path} has more than one channel, only use the first channel')
        depth = nonzero_normalize_depth(depth, midas_mask) # [128,128],[512,512]
        save_tensor2image(depth, os.path.join(opt.workspace, 'depth_resized.jpg'))
        # depth = depth[midas_mask]
        print(f'[INFO] dataset: load depth prompt {depth_path} {depth.shape}')
    else:
        depth = None
        print(f'[WARN] dataset: {depth_path} is not found')
    
    # # load normal
    # normal_path = opt.ref_path.replace('rgba', 'normal')
    # if os.path.exists(normal_path):
    #     normal = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
    #     if normal.shape[-1] == 4:
    #         normal = cv2.cvtColor(normal, cv2.COLOR_BGRA2RGB)
    #     normal = cv2.resize(normal, (opt.w, opt.h), interpolation=cv2.INTER_AREA) 
    #     normal = torch.from_numpy(normal.astype(np.float32) / 255).unsqueeze(0).unsqueeze(0).to(device)
    #     print(f'[INFO] dataset: load normal prompt {normal_path}')
    #     normal = normal[mask]
    # else:
    #     normal = None
    #     print(f'[WARN] dataset: {normal_path} is not found')
    
    model = NeRFNetwork(opt)
    trainer = Trainer(opt.experiment, opt, model, guidance, depth_model=None, 
                        ref_imgs=ref_imgs, ref_depth=depth, 
                        ref_mask=depth_mask, ori_imgs=ori_imgs, 
                        device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=None, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval, scheduler_update_every_step=True)
    

    if opt.test:
        test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=33).dataloader()
        trainer.test(test_loader, write_video=True)
        
        if opt.save_mesh:
            trainer.save_mesh(resolution=256)
            
    else:      
        if opt.use_mvdream:
            opt.batch_size = 4
        train_loader = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, size=100).dataloader(batch_size = opt.batch_size)
        # don't use this
        # trainer.default_view_data = train_loader._data.get_default_view_data()
        valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, size=5).dataloader(batch_size = 1)
        max_epoch = np.ceil(opt.iters / 100).astype(np.int32)
        trainer.train(train_loader, valid_loader, max_epoch)

        # also test
        if opt.final:
            test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=100).dataloader(batch_size = 1)
            trainer.test(test_loader, write_image=False, write_video=True)
        
        if opt.save_mesh:
            trainer.save_mesh(resolution=256)
            
        if opt.refine:
            mv_loader = NeRFDataset(opt, device=device, type='gen_mv', H=opt.H, W=opt.W, size=33).dataloader(batch_size = 1)
            test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=64).dataloader(batch_size = 1)
            trainer.test(mv_loader, save_path=os.path.join(opt.workspace, 'mvimg'), write_image=True, write_video=False)
            if opt.process_mask and not os.path.exists(os.path.join(opt.workspace, 'mask')):
                trainer.mask(os.path.basename(opt.workspace))
            trainer.refine(os.path.join(opt.workspace, 'mvimg'), opt.refine_iters, test_loader)
        