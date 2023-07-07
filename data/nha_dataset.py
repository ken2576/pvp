"""Dataloader for Neural Head Avatar dataset (https://github.com/philgras/neural-head-avatars)."""
import glob
import json
import os
from typing import Text, Optional

import imageio
import numpy as np
from pytorch_lightning import LightningDataModule
import tqdm
import torch
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from data.dataset_fns import crop_face, compute_transform_np, crop_face_deca_bbox, crop_face_deca_warp
from utils import imgHWC_CHW, load_json

class NHADataset(Dataset):
    def __init__(self,
                 scene_path: Text,
                 masked_rgb: bool = True,
                 normalize: bool = True,
                 center_sigma: float = 1.0,
                 xy_sigma: float = 3.0,
                 ):
        """Create a torch.utils.data.Dataset for NHA data

        Args:
            scene_path: Path to the scene
            tracking_path: Path to the tracking data
            num_src_views: Number of source view mesh to use
            masked_rgb: Return masked RGB images
            normalize: Normalize the RGB image to [-1, 1]
        """
        super().__init__()
        self.masked_rgb = masked_rgb

        glob_str = os.path.join(scene_path, 'frame_*', 'image_0000.png')
        self.rgb_paths = sorted(glob.glob(glob_str))

        glob_str = os.path.join(scene_path, 'frame_*', 'masked_0000.png')
        self.masked_rgb_paths = sorted(glob.glob(glob_str))

        glob_str = os.path.join(scene_path, 'frame_*', 'keypoints_static_0000.json')
        json_paths = sorted(glob.glob(glob_str))
        pts2d = []
        for path in tqdm.tqdm(json_paths):
            data = load_json(path)
            person = data['people'][0]
            pts2d.append(np.array(person['face_keypoints_2d']).reshape([-1, 3]))

        assert len(pts2d) == len(self.rgb_paths)

        pts2d = np.array(pts2d)[..., :2]
        cache_path = os.path.join(scene_path, 'cached_data.json')

        if os.path.isfile(cache_path):
            with open(cache_path) as f:
                data = json.load(f)
            # bbox for FFHQ alignment
            c, x, y = [[d[key] for d in data] for key in ['c', 'x', 'y']]
            self.c, self.x, self.y = map(torch.FloatTensor, (c, x, y))
            # bbox for DECA alignment
            dc, ds = [[d[key] for d in data] for key in ['dc', 'ds']]
            self.dc, self.ds = map(torch.FloatTensor, (dc, ds))
        else:
            from data.detectors import FAN
            face_detector = FAN('cpu')
            c, x, y = compute_transform_np(pts2d)
            c = gaussian_filter1d(c, sigma=center_sigma, axis=0)
            x = gaussian_filter1d(x, sigma=xy_sigma, axis=0)
            y = gaussian_filter1d(y, sigma=xy_sigma, axis=0)
            self.c, self.x, self.y = c, x, y

            dc, ds = [], []
            for path in tqdm.tqdm(self.rgb_paths):
                rgb = imageio.imread(path).astype(np.float32) / 255.
                cc, ss = crop_face_deca_bbox(rgb, face_detector)
                dc.append(cc)
                ds.append(ss)
            dc = np.array(dc)
            ds = np.array(ds)
            dc = gaussian_filter1d(dc, sigma=center_sigma, axis=0)
            ds = gaussian_filter1d(ds, sigma=xy_sigma, axis=0)
            self.dc, self.ds = dc, ds

            with open(cache_path, 'w') as f:
                data = [{'c': c[k].tolist(), 'x': x[k].tolist(), 'y': y[k].tolist(),
                         'dc': dc[k].tolist(), 'ds': ds[k].tolist()}
                         for k in range(len(c))]
                json.dump(data, f, ensure_ascii=False, indent=2)

        feats_path = os.path.join(scene_path, 'feats.json')
        with open(feats_path) as f:
            data = json.load(f)
        # bbox for FFHQ alignment
        pose, expr, lmks = [[d[key] for d in data] for key in ['pose', 'expr', 'lmks']]
        self.pose, self.expr, self.lmks= map(torch.FloatTensor, (pose, expr, lmks))

        self.normalize = normalize
        self.usable_idcs = torch.LongTensor([x for x in range(len(self.rgb_paths))])

    def __len__(self):
        return len(self.usable_idcs)

    def __getitem__(self, datum_index):
        index = self.usable_idcs[datum_index].item()
        if self.masked_rgb:
            rgb = imageio.imread(self.masked_rgb_paths[index]).astype(np.float32) / 255.
        else:
            rgb = imageio.imread(self.rgb_paths[index])[..., :3].astype(np.float32) / 255.
        rgb = imgHWC_CHW(torch.FloatTensor(rgb))
        if self.normalize:
            rgb = rgb * 2 - 1

        aligned_rgb = crop_face(rgb[None]+1.,
                                torch.FloatTensor(self.c[index][None]),
                                torch.FloatTensor(self.x[index][None]),
                                torch.FloatTensor(self.y[index][None]),
                                rgb.shape[-2:])[0]-1.

        deca_rgb, tform = crop_face_deca_warp(rgb+1.,
                                              torch.FloatTensor(self.dc[index]),
                                              torch.FloatTensor(self.ds[index]))

        sample = dict(
            rgb=rgb,
            aligned_rgb=aligned_rgb,
            deca_rgb=deca_rgb[0]-1.,
            tform=tform,
            pose=self.pose[index],
            expr=self.expr[index],
            lmks=self.lmks[index],
        )

        return sample

    def filter_data(self, pitch_lim: float, yaw_lim: float):
        self.usable_idcs = [x for x in range(len(self.rgb_paths))]

        pitch_mask = (torch.abs(self.pitch) < pitch_lim).float()
        yaw_mask = (torch.abs(self.yaw) < yaw_lim).float()
        mask = (pitch_mask * yaw_mask).long()
        self.usable_idcs = torch.nonzero(mask)

class NHADataModule(LightningDataModule):
    def __init__(self, 
                 scene_path: Text,
                 train_size: int,
                 test_size: int,
                 masked_rgb: bool = True,
                 normalize: bool = True,
                 batch_size: int = 1,
                 num_workers: int = 0,):
        super().__init__()
        self.scene_path = scene_path
        self.masked_rgb = masked_rgb
        self.batch_size = batch_size
        self.normalized = normalize
        self.num_workers = num_workers
        self.train_size = train_size
        self.test_size = test_size

    def setup(self, stage: str):
        self.nha_full = NHADataset(
                    self.scene_path,
                    self.masked_rgb,
                    self.normalized)
        # Created using indices from 0 to train_size.
        self.nha_train = torch.utils.data.Subset(self.nha_full,
                                                 range(self.train_size))

        # Created using indices from train_size to train_size + test_size.
        self.nha_val = torch.utils.data.Subset(self.nha_full,
                                               range(self.train_size, 
                                                     self.train_size + self.test_size))

    def train_dataloader(self):
        return DataLoader(self.nha_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.nha_val, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.nha_val, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(self.nha_val, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)
