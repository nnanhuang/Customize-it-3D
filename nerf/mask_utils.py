import cv2
import numpy as np
from tqdm import tqdm
import torch
import os
import glob
import shutil
import sys
from natsort import natsorted
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

def is_white_ratio_greater(mask):
    width = mask.shape[0]
    
    left = (width - 64) // 2
    upper = (width - 64) // 2
    right = left + 64
    lower = upper + 64

    middle_region = mask[upper:lower, left:right]

    white_pixels = np.count_nonzero(middle_region)
    black_pixels = 64 * 64 - white_pixels

    white_ratio = white_pixels / (white_pixels + black_pixels)
    black_ratio = black_pixels / (white_pixels + black_pixels)

    return white_ratio > black_ratio

def process_mask(self,load_data_folder,ref_mask,workspace,sam_output,scores_list):   
    # load rendered and SAM generated mask images and process them
    # default view use ground-truth rgb and mask
    device = torch.device('cuda')
    W = self.opt.W
    H = self.opt.H

    # rendered image file path
    mask_files = sorted(glob.glob(load_data_folder+'/*mask.png'))
    depth_files = sorted(glob.glob(load_data_folder+'/*depth.png'))

    # SAM generated mask file path
    sam_folder = sam_output
    sam_mask_files = []
    for filename in os.listdir(sam_folder):
        file_path = os.path.join(sam_folder, filename)
        if os.path.isfile(file_path):
            sam_mask_files.append(file_path)
    sam_mask_files = natsorted(sam_mask_files)

    # for folder_name in os.listdir(sam_folder):
    #     subfolder_path = os.path.join(sam_folder, folder_name)
        
    #     if os.path.isdir(subfolder_path):
    #         file_paths = glob.glob(os.path.join(subfolder_path, '0.png'))

    #         if file_paths:
    #             sam_mask_files.extend(file_paths)
    
    ### now ref img if in the first !!
    ref_num = (len(mask_files)-1) // 2
    # ref_num = 0
    
    output_path = load_data_folder.replace("mvimg", "mask") 
    os.makedirs(output_path, exist_ok=True)

    # process each file
    for i in tqdm(range(len(mask_files)),desc='Processing Masks'):
        score = scores_list[i]
        if i == ref_num :
            result_mask = ref_mask
        elif score < 0.8:
            result_mask = load_depth(mask_files[i]) / 255.0
        else:
            depth = load_depth(depth_files[i]) / 1000.0
            mask_o = load_depth(mask_files[i]) / 255.0
            mask_sam = load_depth(sam_mask_files[i]) / 255.0


            result_mask = cv2.bitwise_or(mask_o, mask_sam)

        cv2.imwrite(os.path.join(output_path, f'{i}.png'),result_mask * 255)
            
def prepare_mask_SAM(self,load_data_folder,reference_path,workspace):
    sam_checkpoint = "mask/sam_vit_h_4b8939.pth"
    model_type = "default"
    device = torch.device('cuda')
    W = self.opt.W
    H = self.opt.H

    # rendered image file path
    rgb_files = sorted(glob.glob(load_data_folder+'/*rgb.png'))
    # rgb_folder = os.path.join('mask',workspace,'rgbs')
    # os.makedirs(rgb_folder, exist_ok=True)
    
    ref_num = (len(rgb_files)-1) // 2
    # ref_num : int = 0
    
    scores_list = []

    # move each file
    for i in tqdm(range(len(rgb_files)), desc="Generating SAM Masks"):
        if i == ref_num:
            # shutil.copy(reference_path, os.path.join(rgb_folder, os.path.basename(rgb_files[i])))
            image = cv2.imread(reference_path, cv2.IMREAD_UNCHANGED)

        else:
            # shutil.copy(rgb_files[i], os.path.join(rgb_folder, os.path.basename(rgb_files[i])))
            image = cv2.imread(rgb_files[i], cv2.IMREAD_UNCHANGED)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (H, W), interpolation=cv2.INTER_AREA) 
        
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        predictor = SamPredictor(sam)
        predictor.set_image(image)

        masks, scores, _ = predictor.predict(multimask_output=False)
        scores_list.append(scores[0])
        # print (f'score is {scores[0]}')

        save_dir = os.path.join('mask',workspace,'sam_masks')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir,f'{i}.png')
        mask = masks[0]
        if not is_white_ratio_greater(mask):
            mask = ~mask
        cv2.imwrite(save_path, mask*255)
        
    return scores_list


def load_depth(file_name):
    import cv2
    D = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    if not D is None and len(D.shape) > 2:
        D = D[:,:,0]
    return    D

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_path', default=None, type=str, help="use image as referance, only support alpha image")
    parser.add_argument('--workspace', default=None, type=str)
    # parser.add_argument('--finish_sam', default=False, type=str)
    # parser.add_argument('--sam_output', default=None, type=str)

    opt = parser.parse_args()

    load_data_folder = os.path.join('results',opt.workspace,'mvimg')
    ref_imgs = cv2.imread(opt.ref_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
    ref_imgs = cv2.cvtColor(ref_imgs, cv2.COLOR_BGRA2RGBA)
    ref_imgs = cv2.resize(ref_imgs, (800, 800), interpolation=cv2.INTER_AREA)

    ref_mask = ref_imgs[:,:,3:] / 255.0
    # cv2.imwrite('mask/res.png',ref_mask*255)


    print(f'[INFO] use SAM to generate masks...')
    # prepare_mask_SAM(load_data_folder, opt.ref_path, opt.workspace)
    print(f'[INFO] Successfully generate masks!')

    sam_output = os.path.join('mask',opt.workspace,'sam_masks')

    print(f'[INFO] composite new masks...')
    process_mask(load_data_folder,ref_mask, opt.workspace, sam_output)
    print(f'[INFO] Successfully composite new masks!')

