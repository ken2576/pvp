import json
import logging
import os
from typing import List, Sequence, Any, Text

import imageio
from PIL import Image
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import torch
import tqdm

def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger

@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.Logger],
) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionaly saves:
    - number of model parameters
    """

    if not trainer.logger:
        return

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["models"] = config["models"]

    # save number of model parameters
    hparams["models/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["models/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["models/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = config["datamodule"]
    hparams["trainer"] = config["trainer"]

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.Logger],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb
            wandb.finish()

def save_results(
    results: List[Any],
    out_dir: Text,
) -> None:
    """Save results to video."""
    os.makedirs(out_dir, exist_ok=True)
    for i, res in tqdm.tqdm(enumerate(results)):
        rgb = res['pred_rgb']
        gt = res['rgb']
        row = torch.cat([rgb, gt], -1)
        to_save = np.clip(
            img_np(imgCHW_HWC(row[0] * 0.5 + 0.5)), 0.0, 1.0) * 255.
        to_save = to_save.astype(np.uint8)
        # write to disk with Pillow
        dst = os.path.join(out_dir, f'{i:05d}.png')
        Image.fromarray(to_save).save(dst)
        # imageio.imsave(os.path.join(out_dir, f'{i:05d}.png'), to_save)

def save_img(
    path,
    image
):
    to_save = np.clip(
        img_np(imgCHW_HWC(image[0] * 0.5 + 0.5)),
        0.0, 1.0) * 255.
    to_save = to_save.astype(np.uint8)
    imageio.imsave(path, to_save)

def imgHWC_CHW(x):
    """Permute image dimension from HWC to CHW."""
    return x.permute([2, 0, 1])

def imgCHW_HWC(x):
    """Permute image dimension from CHW to HWC."""
    return x.permute([1, 2, 0])

def img_np(x):
    return x.detach().cpu().numpy()

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def move_to(obj, device, skip_unknowns=False):
    """Move data to device [cpu|cuda]."""
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device, skip_unknowns)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device, skip_unknowns))
        return res
    else:
        if not skip_unknowns:
            raise TypeError("Invalid type for move_to")
