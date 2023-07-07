import os
from typing import List, Optional

import json
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning import (
    Callback,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import Logger
from models.model import StyleIBRModel
import utils

import sys
sys.path.insert(1, to_absolute_path('third_party/pti')) # Load torch_utils and dnnlib

log = utils.get_logger(__name__)

def get_dataset(config):
    data_module = hydra.utils.instantiate(config.datamodule)
    return data_module()

def train(config: DictConfig) -> Optional[float]:
    """Contains the training pipeline. Can additionally evaluate model on a testset, using best
    weights achieved during training.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """
    # Set up output directory
    if "num" in HydraConfig().get().job:
        out_dir = config.get("out_dir").split('/')
        out_dir.insert(-1, str(HydraConfig().get().job.num))
        out_dir = '/'.join(out_dir)
    else:
        out_dir = config.get("out_dir")
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Convert relative ckpt path to absolute path if necessary
    ckpt_path = config.trainer.get("resume_from_checkpoint")
    if ckpt_path and not os.path.isabs(ckpt_path):
        config.trainer.resume_from_checkpoint = os.path.join(
            hydra.utils.get_original_cwd(), ckpt_path
        )

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule = get_dataset(config)

    # Init lightning model
    log.info(f"Instantiating model <{config.models._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.models)()

    if config['pretrained_ckpt']:
        conf_dict = OmegaConf.to_container(config.models)
        for key in conf_dict:
            if isinstance(conf_dict[key], dict):
                conf_dict[key] = hydra.utils.instantiate(conf_dict[key])
            if isinstance(conf_dict[key], str):
                if '$' in conf_dict[key]:
                    conf_dict[key] = config.models[key]

        model = StyleIBRModel.load_from_checkpoint(config['pretrained_ckpt'],
                                                   strict=False,
                                                   **conf_dict)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

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
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    if config.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)

    # Get metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric and optimized_metric not in trainer.callback_metrics:
        raise Exception(
            "Metric for hyperparameter optimization not found! "
            "Make sure the `optimized_metric` in `hparams_search` config is correct!"
        )
    score = trainer.callback_metrics.get(optimized_metric)

    # Test the model
    if config.get("test"):
        ckpt_path = "best"
        if not config.get("train") or config.trainer.get("fast_dev_run"):
            ckpt_path = None
        log.info("Starting testing!")
        ret = trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

        for idx, r in enumerate(ret):
            with open(os.path.join(log_dir, f"result_{idx}.json"), "w") as f:
                data = json.dumps(r, indent=4)
                f.write(data)

    # Generate videos
    if config.get("predict"):
        ckpt_path = "best"
        if not config.get("train") or config.trainer.get("fast_dev_run"):
            ckpt_path = None
        log.info("Starting rendering!")
        ret = trainer.predict(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        utils.save_results(ret, to_absolute_path(out_dir))

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run") and config.get("train"):
        log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    return score
