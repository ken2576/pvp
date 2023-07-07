import argparse
import glob
import json
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
from scipy.ndimage import gaussian_filter1d

from utils.model_utils import set_requires_grad
from data.dataset_fns import crop_face, compute_transform_np, crop_face_deca_bbox, crop_face_deca_warp
from utils import imgHWC_CHW, load_json
from third_party.DECA.decalib.deca import DECA
from third_party.DECA.decalib.utils.config import cfg as deca_cfg
from third_party.nha.nha.util.lbs import batch_rodrigues


def config_parser():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--folder', type=str,
                        help='path to folder containing the images')
    return parser.parse_args()

def compute_pitch_yaw(R):
    x, y, z = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    yaw = torch.atan2(x, z) * 180 / np.pi
    pitch = torch.atan2(y, z) * 180 / np.pi
    return pitch[:, None], yaw[:, None]

args = config_parser()
device = 'cuda'

center_sigma = 1.0
xy_sigma = 3.0

glob_str = os.path.join(args.folder, 'frame_*', 'image_0000.png')
rgb_paths = sorted(glob.glob(glob_str))

glob_str = os.path.join(args.folder, 'frame_*', 'keypoints_static_0000.json')
json_paths = sorted(glob.glob(glob_str))
pts2d = []
for path in tqdm(json_paths):
    data = load_json(path)
    person = data['people'][0]
    pts2d.append(np.array(person['face_keypoints_2d']).reshape([-1, 3]))

assert len(pts2d) == len(rgb_paths)

pts2d = np.array(pts2d)[..., :2]

cache_path = os.path.join(args.folder, 'cached_data.json')

if os.path.isfile(cache_path):
    with open(cache_path) as f:
        data = json.load(f)
    # bbox for FFHQ alignment
    c, x, y = [[d[key] for d in data] for key in ['c', 'x', 'y']]
    # bbox for DECA alignment
    dc, ds = [[d[key] for d in data] for key in ['dc', 'ds']]
    dc, ds = map(torch.FloatTensor, (dc, ds))
else:
    from data.detectors import FAN
    face_detector = FAN('cuda')
    c, x, y = compute_transform_np(pts2d)
    c = gaussian_filter1d(c, sigma=center_sigma, axis=0)
    x = gaussian_filter1d(x, sigma=xy_sigma, axis=0)
    y = gaussian_filter1d(y, sigma=xy_sigma, axis=0)

    dc, ds = [], []
    for path in tqdm(rgb_paths):
        rgb = imageio.imread(path).astype(np.float32) / 255.
        cc, ss = crop_face_deca_bbox(rgb, face_detector)
        dc.append(cc)
        ds.append(ss)
    dc = np.array(dc)
    ds = np.array(ds)
    dc = gaussian_filter1d(dc, sigma=center_sigma, axis=0)
    ds = gaussian_filter1d(ds, sigma=xy_sigma, axis=0)
    dc, ds = map(torch.FloatTensor, (dc, ds))


    with open(cache_path, 'w') as f:
        data = [{'c': c[k].tolist(), 'x': x[k].tolist(), 'y': y[k].tolist(),
                    'dc': dc[k].tolist(), 'ds': ds[k].tolist()}
                    for k in range(len(c))]
        json.dump(data, f, ensure_ascii=False, indent=2)

deca_cfg.model.use_tex = False
deca_cfg.rasterizer_type = 'pytorch3d'
deca_cfg.model.extract_tex = False
deca = DECA(config = deca_cfg, device=device) # HACK
set_requires_grad(False, deca)
deca.eval()
deca.E_flame.eval()

pose_arr = []
expr_arr = []
lmks_arr = []
for index, path in enumerate(tqdm(rgb_paths)):
    rgb = imageio.imread(path).astype(np.float32) / 255.
    rgb = imgHWC_CHW(torch.FloatTensor(rgb)).to(device)

    deca_rgb, tform = crop_face_deca_warp(rgb,
                                          torch.FloatTensor(dc[index]),
                                          torch.FloatTensor(ds[index]))

    codedict = deca.encode(deca_rgb, False)

    pose = codedict['pose']
    expr = codedict['exp']

    # rot_mat = batch_rodrigues(pose[:, :3])
    # pitch, yaw = compute_pitch_yaw(rot_mat)

    lmks = deca.create_lmks(
        codedict,
        original_image=rgb[None],
        tform=torch.inverse(tform).transpose(1,2)) # HACK ASSUME BATCH=1
    
    pose_arr.append(pose)
    expr_arr.append(expr)
    lmks_arr.append(lmks)

pose_arr = torch.cat(pose_arr, 0).cpu().numpy()
expr_arr = torch.cat(expr_arr, 0).cpu().numpy()
lmks_arr = torch.cat(lmks_arr, 0).cpu().numpy()

filtered_pose = gaussian_filter1d(pose_arr, sigma=0.01, axis=0)
filtered_expr = gaussian_filter1d(expr_arr, sigma=1.0, axis=0)
filtered_lmks = gaussian_filter1d(lmks_arr, sigma=1.0, axis=0)

feats_path = os.path.join(args.folder, 'feats.json')
with open(feats_path, 'w') as f:
    data = [{'pose': p.tolist(), 'expr': e.tolist(), 'lmks': l.tolist()}
                for (p, e, l) in zip(filtered_pose, filtered_expr, filtered_lmks)]
    json.dump(data, f, ensure_ascii=False, indent=2)

rot_vec = torch.FloatTensor(filtered_pose[..., :3])
# Seems to be head_f_world, so tranpose to get world_f_head
rot_mat = batch_rodrigues(rot_vec).permute(0, 2, 1)
pitch, yaw = compute_pitch_yaw(rot_mat)
plt.scatter(yaw, pitch)
plt.savefig(os.path.join(args.folder, 'poses.png'))