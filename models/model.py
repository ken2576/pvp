import os
from typing import Any, Dict, List, Optional, Tuple, Text

from einops import rearrange, repeat
import lpips
import kornia
import numpy as np
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure

from data.dataset_fns import crop_face_deca, gen_mask
from data.detectors import FAN
from models.id_loss import IDLoss
from utils.model_utils import load_generator, set_requires_grad
import utils
from third_party.DECA.decalib.deca import DECA
from third_party.DECA.decalib.utils.config import cfg as deca_cfg
from third_party.nha.nha.util.lbs import batch_rodrigues

log = utils.get_logger(__name__)

# ---------------------------------------------------------------------------- #
#                               Utility Functions                              #
# ---------------------------------------------------------------------------- #

def create_metrics(id_path):
    return {
        "pixel_loss": nn.MSELoss(),
        "lpips_loss":  lpips.LPIPS(net='vgg'),
        "id_loss": IDLoss(id_path),
        "psnr": PeakSignalNoiseRatio(data_range=2.0),
    }

def create_lr_dict(
    latent_lr: float,
):
    return {
        "latent_lr": latent_lr,
    }

def compute_pitch_yaw(R):
    x, y, z = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    yaw = torch.atan2(x, z) * 180 / np.pi
    pitch = torch.atan2(y, z) * 180 / np.pi
    return pitch[:, None], yaw[:, None]

class StyleIBRModel(LightningModule):
    """Implement the IBR model."""

    # ---------------------------------------------------------------------------- #
    #                                 Initializers                                 #
    # ---------------------------------------------------------------------------- #

    def __init__(
        self,
        latent_module_init_fn: Any,
        lr_init_fn: Any,
        optim: Any,
        stylegan_model_path: Text,
        stylegan_ckpt_path: Optional[str],
        id_ckpt_path: Text,
        conditioned_latent: bool,
        use_mystyle: bool,
        **kwargs
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.lr_dict = lr_init_fn()

        # Initialize StyleGAN 2/3
        self.stylegan = load_generator(stylegan_model_path, self.device, False)

        if stylegan_ckpt_path:
            if use_mystyle:
                self.stylegan.synthesis = torch.load(stylegan_ckpt_path).to(self.device) # TODO
            else:
                self.stylegan = torch.load(stylegan_ckpt_path).to(self.device)
            self.stylegan.eval()
            set_requires_grad(False, self.stylegan)

        self.latent_module = latent_module_init_fn(self.stylegan,
                                                   use_mystyle=use_mystyle,
                                                   device=self.device)

        deca_cfg.model.use_tex = False
        deca_cfg.rasterizer_type = 'pytorch3d'
        deca_cfg.model.extract_tex = False
        self.deca = DECA(config = deca_cfg, device='cuda') # HACK
        self.face_detector = FAN('cuda')
        set_requires_grad(False, self.deca)
        self.deca.eval()
        self.deca.E_flame.eval()

        self.optim = optim

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch        
        self.metrics = nn.ModuleDict(create_metrics(id_ckpt_path))
        self.weights = {
            "pixel_loss": kwargs['w_pix_loss'],
            "lpips_loss":  kwargs['w_lpips_loss'],
            "manifold_loss": kwargs['w_manifold'],
            "id_loss": kwargs['w_id_loss'],
            "expr_loss": kwargs['w_expr_loss'],
            "pose_loss": kwargs['w_pose_loss'],
            "rand_id_loss": kwargs['w_rand_id'],
            "consistency_loss": kwargs['w_consistency'],
        }
        self.gamma_pose = kwargs['gamma_pose'] # For scaling the random pose noise
        self.high_res_lpips = kwargs['high_res_lpips']

        self.conditioned_latent = conditioned_latent

    # ---------------------------------------------------------------------------- #
    #                           LightningModule Functions                          #
    # ---------------------------------------------------------------------------- #

    def forward(self, batch: Dict):
        self.stylegan.eval()
        self.metrics['lpips_loss'].eval()
        self.metrics['id_loss'].eval()
        self.deca.eval() # HOLYSHIT PYTORCH LIGHTNING IS HOT GARBAGE
        feats, pitch, yaw, pose, expr, masks, tforms = self.prepare_batch(batch)
        ret = self.render_rgba(feats, pitch, yaw)
        pred_expr, gt_expr, pred_pose, gt_pose, rand_pred, expr_delta = self.render_random_pose(
            pitch, yaw, pose, expr, tforms)
        ret['pred_expr'] = pred_expr
        ret['gt_expr'] = gt_expr
        ret['pred_pose'] = pred_pose
        ret['gt_pose'] = gt_pose
        ret['rand_pred'] = rand_pred
        ret['expr_delta'] = expr_delta
        ret['lmk_mask'] = masks

        return ret

    def training_step(self, batch: Dict, batch_idx: int):
        ret = self.forward(batch=batch)
        loss, log_dict = self.calculate_loss(batch, ret, 'train')
        self.write_to_log(log_dict, 'train', on_step=True, on_epoch=True)

        if self.trainer.global_step % 100 == 0:
            pred = ret['pred_rgb'].detach()
            gt = batch['aligned_rgb']
            grid = torch.cat([pred, gt], -1)[0] * 0.5 + 0.5
            grid = torch.clip(grid, 0.0, 1.0)
            self.logger.experiment.add_image('train/pred_gt', grid, self.trainer.global_step)

        return {"loss": loss}

    def test_step(self, batch: Dict, batch_idx: int):
        ret = self.forward(batch=batch)
        loss, log_dict = self.calculate_loss(batch, ret, 'test')
        self.write_to_log(log_dict, 'test', on_step=True, on_epoch=True)

        return {"loss": loss}

    def validation_step(self, batch: Dict, batch_idx: int):
        ret = self.forward(batch=batch)
        loss, log_dict = self.calculate_loss(batch, ret, 'val')
        self.write_to_log(log_dict, 'validation', on_step=True, on_epoch=True)

        if batch_idx == 0:
            pred = ret['pred_rgb'].detach()
            gt = batch['aligned_rgb']
            grid = torch.cat([pred, gt], -1)[0] * 0.5 + 0.5
            grid = torch.clip(grid, 0.0, 1.0)
            self.logger.experiment.add_image('val/pred_gt', grid, self.trainer.global_step)

            if 'rand_pred' in ret:
                tmp = ret['rand_pred'].detach()[0] * 0.5 + 0.5
                tmp = torch.clip(tmp, 0.0, 1.0)
                self.logger.experiment.add_image('val/rand_pred', tmp, self.trainer.global_step)

        return {"loss": loss}

    def predict_step(self, batch: Dict, batch_idx: int):
        output = self.forward(batch=batch)
        sample = {
            'pred_rgb': output['pred_rgb'],
            'rgb': batch['aligned_rgb'],
        } 
        return sample

    def on_train_start(self):
        # Setup noise inputs.
        noise_bufs = {
            name: buf for (name, buf) in self.stylegan.synthesis.named_buffers()
            if 'noise_const' in name}
        # Init noise.
        for buf in noise_bufs.values():
            buf[:].data = torch.randn_like(buf)
            buf.requires_grad = True

    def on_train_end(self):
        # Setup noise inputs.
        noise_bufs = {
            name: buf for (name, buf) in self.stylegan.synthesis.named_buffers()
            if 'noise_const' in name}
        for buf in noise_bufs.values():
            buf.requires_grad = False

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                    optimizer_closure, on_tpu, using_native_amp, using_lbfgs):

        optimizer.step(closure=optimizer_closure)

    def configure_optimizers(self):
        lr = self.lr_dict

        # Setup noise inputs.
        noise_bufs = {
            name: buf for (name, buf) in self.stylegan.synthesis.named_buffers()
            if 'noise_const' in name}

        params = [
            {"params": self.latent_module.parameters()},
            {"params": list(noise_bufs.values())},
        ]
        latent_optim = self.optim(params, lr=lr['latent_lr']) # TODO setup ADAM params

        return [
            {"optimizer": latent_optim}
        ]

    def load_latent(self, path: str):
        self.latent_module.load_ckpt(path)

    def write_to_log(self, log_dict: Dict, mode: str, **log_argv):
        for key in log_dict:
            self.log(f"{key}", log_dict[key], **log_argv)

    # ---------------------------------------------------------------------------- #
    #                               Rendering Helpers                              #
    # ---------------------------------------------------------------------------- #

    def render_rgba(self, feats, pitch, yaw):
        """Render target viewpoint via texture mapping from the source viewpoints.
        
        Args:
            feats: features for expression encoder
            pitch: head pitch in degrees
            yaw: head yaw in degrees
        Returns:
            Rendered images.
        """

        if self.conditioned_latent:
            ws, w_delta = self.latent_module.generate(feats,
                                                      pitch/180*np.pi,
                                                      yaw/180*np.pi)
        else:
            ws, w_delta = self.latent_module.generate(pitch/180*np.pi,
                                                      yaw/180*np.pi)

        b, v = ws.shape[:2]
        if ws.shape[2] == 1:
            ws = repeat(ws, 'b 1 1 c -> b n c', n=self.stylegan.num_ws)
        else:
            ws = rearrange(ws, 'b 1 n c -> b n c')
        forward_fn = lambda x: self.stylegan.synthesis(x,
                                                       noise_mode='const',
                                                       force_fp32=True)

        # style_pred = checkpoint(forward_fn, ws)
        style_pred = forward_fn(ws)
        lr_style_pred = F.interpolate(style_pred, (512, 512),
                                      mode='bilinear', align_corners=True,
                                      antialias=True)

        return {
            'style_pred': style_pred,
            'pred_rgb': lr_style_pred,
            'w_delta': w_delta
        }

    def render_random_pose(self, pitch, yaw, pose, expr, tforms):
        """Render random target pose via deca.
        
        Args:
            pitch: head pitch in degrees
            yaw: head yaw in degrees
            pose: pose parameters
            expr: expression parameters
            tforms: transforms for cropping
        Returns:
            Rendered images.
        """
        gt_expr = expr + \
            torch.randn_like(expr) * self.gamma_pose

        feats = torch.cat([pose[:, 3:], gt_expr], -1)

        if self.conditioned_latent:
            ws, w_delta = self.latent_module.generate(feats,
                                                      pitch/180*np.pi,
                                                      yaw/180*np.pi)
        else:
            ws, w_delta = self.latent_module.generate(pitch/180*np.pi,
                                                      yaw/180*np.pi)

        b, v = ws.shape[:2]
        if ws.shape[2] == 1:
            ws = repeat(ws, 'b 1 1 c -> b n c', n=self.stylegan.num_ws)
        else:
            ws = rearrange(ws, 'b 1 n c -> b n c')
        forward_fn = lambda x: self.stylegan.synthesis(x,
                                                       noise_mode='const',
                                                       force_fp32=True)

        # style_pred = checkpoint(forward_fn, ws)
        style_pred = forward_fn(ws)

        crops = []
        for x, tform in zip(style_pred, tforms):
            # crop, _ = crop_face_deca(utils.imgCHW_HWC(x*0.5+0.5), self.face_detector)
            crop = kornia.geometry.warp_perspective((x*0.5+0.5)[None], tform,
                                                    (224, 224), align_corners=False)
            crops.append(crop)

        crops = torch.cat(crops, 0)
        newdict = self.deca.encode(crops, False)
        pred_expr = newdict['exp']

        pred_pose = newdict['pose'][:, :3] # the head pose should be the same
        gt_pose = pose[:, :3]

        return pred_expr, gt_expr, pred_pose, gt_pose, style_pred, w_delta

    # ---------------------------------------------------------------------------- #
    #                                Loss Functions                                #
    # ---------------------------------------------------------------------------- #

    def calculate_loss(self, data: Dict, output: Dict, mode: str = 'train'):
        pred = output['pred_rgb']
        rand_pred = output['rand_pred']
        gt = data['aligned_rgb']
        lmk_mask = 1. - output['lmk_mask'] # invert the mask

        if self.high_res_lpips:
            lpips_pred = pred
            lpips_gt = gt
        else:
            lpips_pred = F.interpolate(pred, (256, 256),
                                    align_corners=True, mode='bilinear',
                                    antialias=True)
            lpips_gt = F.interpolate(gt, (256, 256),
                                    align_corners=True, mode='bilinear',
                                    antialias=True)

        id_loss_val = self.metrics['id_loss'](pred, gt)
        pix_loss_val = self.metrics['pixel_loss'](pred, gt)
        lpips_loss_val = self.metrics['lpips_loss'](lpips_pred, lpips_gt)
        psnr_val = self.metrics['psnr'](pred, data['aligned_rgb'])

        rand_pred = F.interpolate(rand_pred, (512, 512),
                                  align_corners=True, mode='bilinear',
                                  antialias=True)

        consistency_loss = torch.mean(
            ((pred - rand_pred) * lmk_mask) ** 2
        )

        # Regularize manifold encoder
        w_delta = output['w_delta']
        if w_delta is not None:
            manifold_loss = torch.mean(w_delta ** 2)
        else:
            manifold_loss = 0.0
        expr_delta = output['expr_delta']
        if expr_delta is not None:
            manifold_loss += torch.mean(expr_delta ** 2)

        expr_loss = torch.mean((output['pred_expr'] - output['gt_expr'])**2)
        pose_loss = torch.mean((output['pred_pose'] - output['gt_pose'])**2)

        if self.weights['rand_id_loss'] > 0.0:
            rand_id_loss = self.metrics['id_loss'](rand_pred, pred)
        else:
            rand_id_loss = 0.0


        log_dict = {
            f'{mode}/lpips': lpips_loss_val,
            f'{mode}/pixel_loss': pix_loss_val,
            f'{mode}/psnr': psnr_val,
            f'{mode}/manifold_loss': manifold_loss,
            f'{mode}/id_loss': id_loss_val,
            f'{mode}/expr_loss': expr_loss,
            f'{mode}/pose_loss': pose_loss,
            f'{mode}/rand_id_loss': rand_id_loss,
            f'{mode}/consistency_loss': consistency_loss,
        }

        loss = pix_loss_val * self.weights['pixel_loss'] + \
            lpips_loss_val * self.weights['lpips_loss'] + \
            manifold_loss * self.weights['manifold_loss'] + \
            id_loss_val * self.weights['id_loss'] + \
            expr_loss * self.weights['expr_loss'] + \
            pose_loss * self.weights['pose_loss'] + \
            rand_id_loss * self.weights['rand_id_loss'] + \
            consistency_loss * self.weights['consistency_loss']

        if mode == 'val' or mode == 'test':
            log_dict[f'{mode}/ssim'] = structural_similarity_index_measure(
                pred*0.5+0.5, gt*0.5+0.5, data_range=1.0)

        log_dict[f'{mode}/loss'] = loss

        return loss, log_dict

    # ---------------------------------------------------------------------------- #
    #                                   Utilities                                  #
    # ---------------------------------------------------------------------------- #

    def prepare_batch(self, batch):

        crops = batch['deca_rgb']
        tforms = batch['tform']
        with torch.no_grad():
            if 'pose' not in batch or 'expr' not in batch or 'lmks' not in batch:
                codedict = self.deca.encode(crops*0.5+0.5, False)

                pose = codedict['pose']
                expr = codedict['exp']

                lmks = self.deca.create_lmks(
                    codedict,
                    original_image=batch['rgb'],
                    tform=torch.inverse(tforms[0]).transpose(1,2)) # HACK ASSUME BATCH=1
            else:
                pose = batch['pose']
                expr = batch['expr']
                lmks = batch['lmks']

            rot_mat = batch_rodrigues(pose[:, :3])
            pitch, yaw = compute_pitch_yaw(rot_mat)
        masks = gen_mask(lmks, batch['rgb'].shape[-2:])

        feature = torch.cat([pose[:, 3:], expr], -1)
        return feature, pitch, yaw, pose, expr, masks, tforms
