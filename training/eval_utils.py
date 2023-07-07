import os
from typing import List, Optional

import json
import torch
import numpy as np
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm
from pytorch_lightning import (
    Callback,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import Logger
import utils
from models.model import StyleIBRModel

import sys
sys.path.insert(1, to_absolute_path('third_party/pti')) # Load torch_utils and dnnlib

log = utils.get_logger(__name__)

def get_dataset(config):
    data_module = hydra.utils.instantiate(config.datamodule)
    return data_module()

@torch.no_grad()
def test(config: DictConfig) -> Optional[float]:
    """Contains the training pipeline. Can additionally evaluate model on a testset, using best
    weights achieved during training.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """
    # Set up output directory
    out_dir = to_absolute_path(config.get("out_dir"))
    os.makedirs(out_dir, exist_ok=True)
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Convert relative ckpt path to absolute path if necessary
    ckpt_path = config.get("ckpt_path")
    if ckpt_path and not os.path.isabs(ckpt_path):
        config.trainer.resume_from_checkpoint = os.path.join(
            hydra.utils.get_original_cwd(), ckpt_path
        )

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule = get_dataset(config)

    # Init lightning model
    log.info(f"Instantiating model <{config.models._target_}>")
    conf_dict = OmegaConf.to_container(config.models)
    for key in conf_dict:
        if isinstance(conf_dict[key], dict):
            conf_dict[key] = hydra.utils.instantiate(conf_dict[key])
    model = StyleIBRModel.load_from_checkpoint(ckpt_path, **conf_dict)

    # Init lightning loggers
    logger: List[Logger] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        callbacks=None,
        datamodule=datamodule,
        trainer=trainer,
        logger=logger,
    )

    # ---------------------------------------------------------------------------- #
    #                               Evaluate Metrics                               #
    # ---------------------------------------------------------------------------- #

    # Test the model
    log.info("Starting testing!")
    ret = trainer.test(model=model, datamodule=datamodule)

    for idx, r in enumerate(ret):
        with open(os.path.join(out_dir, f"result_{idx}.json"), "w") as f:
            data = json.dumps(r, indent=4)
            f.write(data)

    # ---------------------------------------------------------------------------- #
    #                             Render Video Sequence                            #
    # ---------------------------------------------------------------------------- #

    # Generate videos
    log.info("Starting rendering!")
    ret = trainer.predict(model=model, datamodule=datamodule)

    save_dir = os.path.join(out_dir, 'video')
    utils.save_results(ret, save_dir)
    os.system(f'ffmpeg -y -r 24 -i "{save_dir}/%05d.png" -vcodec h264 -pix_fmt yuv420p -crf 8 {out_dir}/output.mp4')

    datamodule.teardown(stage="test")

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=None,
        logger=logger,
    )
