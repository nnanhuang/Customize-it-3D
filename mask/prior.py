import cv2
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import os

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
    parser.add_argument('--finish_sam', default=False, type=str)
    parser.add_argument('--sam_output', default=None, type=str)

    opt = parser.parse_args()
    
    sam_checkpoint = "mask/sam_vit_h_4b8939.pth"
    device = "cuda"
    model_type = "default"

    image = cv2.imread('mask/test/df_ep0100_0000_rgb.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask_path = 'results/chair400/mvimg/df_ep0100_0000_mask.png'
    mask_prompt = load_depth(mask_path) / 255
    mask_prompt = cv2.resize(mask_prompt,(256,256))

    # print(mask_prompt.shape) # [1,256,256],lower resolution mask
    mask_prompt = 1 - mask_prompt
    threshold_value = 0.9
    _, mask_prompt = cv2.threshold(mask_prompt, threshold_value, 1, cv2.THRESH_BINARY)
    cv2.imwrite('mask/changed_mask.png', mask_prompt*255)
    
    mask_prompt = mask_prompt[None, :, :]

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    predictor.set_image(image)

    # input_point= np.array([[400, 400]])
    # input_label= np.array([1])

    masks, _, _ = predictor.predict(
        # point_coords = input_point,
        # point_labels = input_label,
        mask_input=mask_prompt,
        multimask_output=False)
    # print(masks.shape) # [1,800,800]
    # print(masks) # true/false

    save_path = 'mask/res.png'
    mask = masks[0]
    cv2.imwrite(save_path, mask*255)
