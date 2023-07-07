import os
import glob
import argparse
from PIL import Image
import cv2
import skimage
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from utils import imgHWC_CHW, imgCHW_HWC, img_np
from third_party.modnet.src.models.modnet import MODNet

def config_parser():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--folder', type=str,
                        help='path to the image folder')
    parser.add_argument('--use_modnet', action='store_true',
                        help='use MODNet to generate the mask')
    parser.add_argument('--modnet_ckpt', type=str,
                        default='/home/ken/projects/monostylegan/assets/modnet_photographic_portrait_matting.ckpt',
                        help='path to the MODNet checkpoint')
    return parser.parse_args()

def get_mask(img):
    # convert to LAB
    lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)

    # extract A channel
    A = lab[:,:,1]

    # threshold A channel
    thresh = cv2.threshold(A, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    # blur threshold image
    blur = cv2.GaussianBlur(thresh, (0,0), sigmaX=1.0, sigmaY=1.0, borderType = cv2.BORDER_DEFAULT)

    # stretch so that 255 -> 255 and 127.5 -> 0
    mask = skimage.exposure.rescale_intensity(blur, in_range=(127.5,255), out_range=(0,255)).astype(np.uint8)

    return mask

if __name__ == '__main__':
    args = config_parser()
    img_paths = sorted(glob.glob(
        os.path.join(args.folder, '*', 'image_0000.png')))
    
    device = 'cuda'

    # Load MODNet
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet).to(device)
    weights = torch.load(args.modnet_ckpt)
    modnet.load_state_dict(weights)
    modnet = modnet.eval()

    # Read images and save them as a video
    for path in tqdm(img_paths):
        rgb = Image.open(path)
        rgb = np.array(rgb) / 255.

        if args.use_modnet:
            rgb_tensor = imgHWC_CHW(torch.FloatTensor(rgb))[None]
            _, _, mask = modnet(rgb_tensor*2-1., True)
            modnet_mask = mask.squeeze().detach().cpu().numpy()

            mask_path = os.path.join(os.path.split(path)[0], 'seg_0000.png')
            mask = np.array(Image.open(mask_path))
            mask = (mask > 0).astype(np.float32)
            mask *= modnet_mask
        else:
            mask_path = os.path.join(os.path.split(path)[0], 'seg_0000.png')
            mask = np.array(Image.open(mask_path))
            mask = (mask > 0).astype(np.float32)

            rgb_cv2 = cv2.imread(path)
            cv2_mask = get_mask(rgb_cv2) / 255.
            mask *= cv2_mask
        
        rgb = rgb[..., :3] * mask[..., None] + 0 * (1-mask[..., None])

        rgb = rgb * 255.
        rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
        # imageio.imsave(dst, crop)
        
        out_folder = os.path.split(path)[0]
        dst = os.path.join(out_folder, "masked_0000.png")
        Image.fromarray(rgb).save(dst)
        exit()
