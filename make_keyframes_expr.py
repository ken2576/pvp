import argparse
import glob
import imageio
import os
import json
import tqdm
import torch
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.cluster.vq import vq, kmeans, whiten
from data.dataset_fns import crop_region, crop_face
from data.dataset_fns import compute_transform_np
from utils import load_json, imgHWC_CHW, imgCHW_HWC, img_np
from third_party.nha.nha.util.lbs import batch_rodrigues

def config_parser():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--folder', type=str,
                        help='path to the image folder')
    parser.add_argument('--out_folder', type=str,
                        help='path to the output folder')
    parser.add_argument('--n_cluster', type=int,
                        help="how many clusters")
    parser.add_argument('--n_sample', type=int,
                        help='how many samples per cluster')
    parser.add_argument('--val_start', type=int,
                        help='starting frame of the validation data')
    parser.add_argument('--masked_rgb', action='store_true',
                        help='mask out the RGB')
    return parser.parse_args()

def compute_pitch_yaw(R):
    x, y, z = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    yaw = torch.atan2(x, z) * 180 / np.pi
    pitch = torch.atan2(y, z) * 180 / np.pi
    return pitch[:, None], yaw[:, None]

args = config_parser()
center_sigma = 1.0
xy_sigma = 3.0

feats_path = os.path.join(args.folder, 'feats.json')
with open(feats_path) as f:
    data = json.load(f)
# bbox for FFHQ alignment
pose, expr, lmks = [[d[key] for d in data] for key in ['pose', 'expr', 'lmks']]
pose, expr, lmks= map(torch.FloatTensor, (pose, expr, lmks))


rot_vec = pose[..., :3]
# Seems to be head_f_world, so tranpose to get world_f_head
rot_mat = batch_rodrigues(rot_vec).permute(0, 2, 1)
pitch, yaw = compute_pitch_yaw(rot_mat)

poses = torch.cat([pitch, yaw, expr], -1)
poses = poses[:args.val_start]
poses = poses.numpy()

print(poses.shape)

whitened = whiten(poses)
codebook, _ = kmeans(whitened, args.n_cluster)
cluster_ids, _ = vq(whitened, codebook)

all_indices = []
# Randomly pick an index in each cluster
for c in set(cluster_ids):
    cluster_indices = np.where(cluster_ids == c)[0]
    if len(cluster_indices) > 0:
        random_index = np.random.choice(cluster_indices, args.n_sample, replace=False)
        all_indices.extend(random_index)

all_indices = sorted(all_indices)

img_paths = sorted(glob.glob(
    os.path.join(args.folder, 'frame_*', 'image_0000.png')))

glob_str = os.path.join(args.folder, 'frame_*', 'keypoints_static_0000.json')
json_paths = sorted(glob.glob(glob_str))

pts2d = []
for path in tqdm.tqdm(json_paths):
    data = load_json(path)
    person = data['people'][0]
    pts2d.append(np.array(person['face_keypoints_2d']).reshape([-1, 3]))

pts2d = np.array(pts2d)[..., :2]
c, x, y = compute_transform_np(pts2d)

c = torch.FloatTensor(gaussian_filter1d(c, sigma=center_sigma, axis=0))
x = torch.FloatTensor(gaussian_filter1d(x, sigma=xy_sigma, axis=0))
y = torch.FloatTensor(gaussian_filter1d(y, sigma=xy_sigma, axis=0))

os.makedirs(args.out_folder, exist_ok=True)
for new_id, idx in enumerate(all_indices):
    print(img_paths[idx])
    data = load_json(json_paths[idx])
    person = data['people'][0]
    pts2d = torch.FloatTensor(person['face_keypoints_2d']).reshape([-1, 3])

    dst = os.path.join(args.out_folder, f"{new_id:04d}.png")
    # shutil.copy2(img_paths[idx], dst)
    rgb = imageio.imread(img_paths[idx]).astype(np.float32) / 255.
    if args.masked_rgb:
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3] * rgb[..., -1:]
        else:
            frame_idx = img_paths[idx].split(os.sep)[-2]
            masked_rgb_path = os.path.join(args.folder, frame_idx, 'masked_0000.png')
            mask_path = os.path.join(args.folder, frame_idx, 'seg_0000.png')
            if os.path.exists(masked_rgb_path):
                rgb = imageio.imread(masked_rgb_path).astype(np.float32) / 255.
            else:
                mask = imageio.imread(mask_path).astype(np.float32) / 255.
                mask = (mask > 0).astype(np.float32)
                rgb = rgb[..., :3] * mask[..., None] + 0 * (1-mask[..., None])
    rgb = imgHWC_CHW(torch.FloatTensor(rgb))
    # crop = crop_region(rgb[None], pts2d[None, ..., :2], (1024, 1024))[0] # TODO smooth with Gaussian?
    crop = crop_face(rgb[None], c[idx][None],
                     x[idx][None], y[idx][None],
                     (1024, 1024), padding_mode='border')[0]

    crop = img_np(imgCHW_HWC(crop)) * 255.
    crop = np.clip(crop, 0.0, 255.0).astype(np.uint8)
    # imageio.imsave(dst, crop)
    Image.fromarray(crop).save(dst)

