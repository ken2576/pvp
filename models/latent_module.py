import os
import glob
from typing import Text, Optional

from einops import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_utils import compute_w_stats


# ---------------------------------------------------------------------------- #
#                           Latent Module in W+ Space                          #
# ---------------------------------------------------------------------------- #

class WLatent(nn.Module):
    """Implement the module to record W latent code."""

    def __init__(
        self,
        stylegan: nn.Module,
        n_samples: int = 10000,
        seed: int = 124):
        super().__init__()
        # Compute mean latent code
        w_avg, w_std = compute_w_stats(stylegan, n_samples,
                                   seed=seed)
        # self.w_opt = nn.Parameter(w_avg.clone().detach().requires_grad_(True))
        self.register_buffer('w_opt', w_avg.expand([-1, 18, -1]).clone().detach().requires_grad_(False), persistent=True)
        sh = (1, 1, 512)
        self.w_pitch = nn.Parameter(torch.zeros(sh, device=w_avg.device).requires_grad_(True))
        self.w_yaw = nn.Parameter(torch.zeros(sh, device=w_avg.device).requires_grad_(True))
        self.register_buffer('w_avg', w_avg, persistent=False)
        self.register_buffer('w_std', w_std, persistent=False)

    def set_latent(self, x):
        # self.w_opt = nn.Parameter(x.clone().detach().requires_grad_(True))
        self.register_buffer('w_opt', x.clone().detach().requires_grad_(False), persistent=True)

    @property
    def id_latent(self):
        return self.w_opt

    def generate(self, pitch: float, yaw: float, **kwargs):
        ws = (self.w_opt[:, None] +
              self.w_pitch[:, None] * pitch +
              self.w_yaw[:, None] * yaw)
        return ws, None

class ExprWLatent(WLatent):
    """Implement the module to record W latent code.
    Additionally, it has an MLP to represent expression changes.
    """

    def __init__(
        self,
        stylegan: nn.Module,
        n_samples: int = 10000,
        seed: int = 124,
        input_size: int = 109,
        hidden_size: int = 128,
        latent_size: int = 512,
        layer_count: int = 2,
        mod_style_count: int = 8,
        nonlinearity: Text = 'lrelu',
        pti_path: Optional[Text] = None,
        **kwargs
        ):
        super().__init__(stylegan, n_samples, seed)

        self.style_count = stylegan.num_ws
        self.mod_style_count = mod_style_count

        act_fn = None
        if nonlinearity == 'lrelu':
            act_fn = nn.LeakyReLU(inplace=True)
        elif nonlinearity == 'relu':
            act_fn = nn.ReLU(inplace=True)
        else:
            raise NotImplementedError

        if pti_path:
            w_opt = torch.load(pti_path)
            self.set_latent(w_opt)

        self.networks = nn.ModuleList()
        for _ in range(mod_style_count):
            layers = []
            layers.append(nn.Linear(input_size, hidden_size, bias=True))
            layers.append(act_fn)
            for _ in range(layer_count-2):
                layers.append(nn.Linear(hidden_size, hidden_size, bias=True))
                layers.append(act_fn)
            layers.append(nn.Linear(hidden_size, latent_size, bias=True))
            self.networks.append(nn.Sequential(*layers))

    @property
    def id_latent(self):
        if self.w_opt.shape[1] == 1:
            return self.w_opt.expand(-1, self.style_count, -1)
        else:
            return self.w_opt

    def generate(self, x: torch.Tensor, pitch: float, yaw: float,
                 return_x_only=False, **kwargs):

        if self.w_opt.shape[1] == 1:
            w_plus = repeat(self.w_opt,
                            'b 1 c -> b w c',
                            w=self.style_count)
        else:
            w_plus = self.w_opt
        dynamic_latent, static_latent = w_plus.split( # TODO invert this?
            (self.mod_style_count, self.style_count-self.mod_style_count), -2)

        delta = torch.stack([network(x) for network in self.networks], 1)
        if (self.style_count - self.mod_style_count) != 0:
            x = delta + dynamic_latent
            x = torch.cat([x, static_latent], 1)
        else:
            x = delta + dynamic_latent

        ws = (x[:, None] +
              self.w_pitch[:, None] * pitch +
              self.w_yaw[:, None] * yaw)

        if return_x_only:
            return x[:, None]

        return ws, delta

class ExprWPlusLatent(nn.Module):
    """Implement the module to record W plus latent code.
    Additionally, it has an MLP to represent expression changes.
    """

    def __init__(
        self,
        stylegan: nn.Module,
        n_samples: int = 10000,
        seed: int = 124,
        input_size: int = 109,
        hidden_size: int = 128,
        latent_size: int = 512,
        layer_count: int = 2,
        mod_style_count: int = 8,
        nonlinearity: Text = 'lrelu',
        pti_path: Optional[Text] = None,
        **kwargs
        ):
        super().__init__()
        # Compute mean latent code
        w_avg, w_std = compute_w_stats(stylegan, n_samples,
                                   seed=seed)
        self.register_buffer('w_opt', w_avg.expand([-1, 18, -1]).clone().detach().requires_grad_(False), persistent=True)
        sh = (1, 18, 512)
        self.w_pitch = nn.Parameter(torch.zeros(sh, device=w_avg.device).requires_grad_(True))
        self.w_yaw = nn.Parameter(torch.zeros(sh, device=w_avg.device).requires_grad_(True))
        self.register_buffer('w_avg', w_avg, persistent=False)
        self.register_buffer('w_std', w_std, persistent=False)

        self.style_count = stylegan.num_ws
        self.mod_style_count = mod_style_count

        act_fn = None
        if nonlinearity == 'lrelu':
            act_fn = nn.LeakyReLU(inplace=True)
        elif nonlinearity == 'relu':
            act_fn = nn.ReLU(inplace=True)
        else:
            raise NotImplementedError
        
        if pti_path:
            w_opt = torch.load(pti_path)
            self.set_latent(w_opt)

        self.networks = nn.ModuleList()
        for _ in range(mod_style_count):
            layers = []
            layers.append(nn.Linear(input_size, hidden_size, bias=True))
            layers.append(act_fn)
            for _ in range(layer_count-2):
                layers.append(nn.Linear(hidden_size, hidden_size, bias=True))
                layers.append(act_fn)
            layers.append(nn.Linear(hidden_size, latent_size, bias=True))
            self.networks.append(nn.Sequential(*layers))

    def set_latent(self, x):
        # self.w_opt = nn.Parameter(x.clone().detach().requires_grad_(True))
        self.register_buffer('w_opt', x.clone().detach().requires_grad_(False), persistent=True)

    @property
    def id_latent(self):
        if self.w_opt.shape[1] == 1:
            return self.w_opt.expand(-1, self.style_count, -1)
        else:
            return self.w_opt

    def generate(self, x: torch.Tensor, pitch: float, yaw: float,
                 return_x_only=False, **kwargs):

        if self.w_opt.shape[1] == 1:
            w_plus = repeat(self.w_opt,
                            'b 1 c -> b w c',
                            w=self.style_count)
        else:
            w_plus = self.w_opt
        dynamic_latent, static_latent = w_plus.split( # TODO invert this?
            (self.mod_style_count, self.style_count-self.mod_style_count), -2)

        delta = torch.stack([network(x) for network in self.networks], 1)
        if (self.style_count - self.mod_style_count) != 0:
            x = delta + dynamic_latent
            x = torch.cat([x, static_latent], 1)
        else:
            x = delta + dynamic_latent

        ws = (x[:, None] +
              self.w_pitch[:, None] * pitch +
              self.w_yaw[:, None] * yaw)

        if return_x_only:
            return x[:, None]

        return ws, delta

# ---------------------------------------------------------------------------- #
#                         Latent Module in Alpha Space                         #
# ---------------------------------------------------------------------------- #

class AlphaLatent(nn.Module):
    """Implement the module to record latent code in alpha-space.
    """
    def __init__(
        self,
        stylegan: nn.Module,
        anchor_dir: Text,
        device: Text,
        n_samples: int = 10000,
        seed: int = 124,
        beta: float = 0.2, #TODO check
        **kwargs,
        ):
        super().__init__()
        # Compute mean latent code
        w_avg, w_std = compute_w_stats(stylegan, n_samples,
                                   seed=seed)

        # Compute mean latent code from the anchors
        paths = glob.glob(os.path.join(anchor_dir, '*.pt'))
        all_anchors = [torch.load(x).to(device) for x in paths]
        all_anchors = torch.cat(all_anchors, 0)
        self.register_buffer('all_anchors', all_anchors, persistent=True)
        alpha = torch.ones([len(self.all_anchors), 1])
        alpha = alpha / len(self.all_anchors)
        w_opt = self.alpha_to_w(alpha)
        self.beta = beta

        # self.w_opt = nn.Parameter(w_avg.clone().detach().requires_grad_(True))
        self.register_buffer('w_opt', w_opt.clone().detach().requires_grad_(False), persistent=True)
        self.register_buffer('alpha', alpha.clone().detach().requires_grad_(False), persistent=True)
        self.w_pitch = nn.Parameter(torch.zeros_like(alpha).requires_grad_(True))
        self.w_yaw = nn.Parameter(torch.zeros_like(alpha).requires_grad_(True))
        self.register_buffer('w_avg', w_avg, persistent=False)
        self.register_buffer('w_std', w_std, persistent=False)

    @property
    def id_latent(self):
        return self.w_opt

    def alpha_to_w(self, alpha):
        ws = torch.einsum('ij...,i...->j...', alpha, self.all_anchors)
        return ws

    def get_constrained_alpha(self, alphas):
        if self.beta is not None:
            alphas = alphas + self.beta
            alphas = F.softplus(alphas, beta=100)
            alphas = alphas - self.beta

        alphas = alphas / torch.sum(alphas, dim=0, keepdim=True)

        return alphas

    def generate(self, pitch, yaw, w_noise=None):
        alpha = (self.alpha[:, None] +
                 self.w_pitch[:, None] * pitch +
                 self.w_yaw[:, None] * yaw)
        ws = self.alpha_to_w(self.get_constrained_alpha(alpha))
        ws = ws[:, None]
        if w_noise is not None:
            ws += w_noise[:, None]
        return ws, None


class ExprAlpha(AlphaLatent):
    """Implement the module to record latent code in alpha-space.
    Additionally, it has an MLP to represent expression changes.
    """

    def __init__(
        self,
        stylegan: nn.Module,
        anchor_dir: Text,
        device: Text,
        n_samples: int = 10000,
        seed: int = 124,
        input_size: int = 115,
        hidden_size: int = 128,
        latent_size: int = 512,
        layer_count: int = 2,
        mod_style_count: int = 8,
        nonlinearity: Text = 'lrelu',
        ):
        super().__init__(stylegan, anchor_dir, device, n_samples, seed)

        self.style_count = stylegan.num_ws
        self.mod_style_count = mod_style_count

        act_fn = None
        if nonlinearity == 'lrelu':
            act_fn = nn.LeakyReLU(inplace=True)
        elif nonlinearity == 'relu':
            act_fn = nn.ReLU(inplace=True)
        else:
            raise NotImplementedError

        self.networks = nn.ModuleList()
        # input_size = len(self.all_anchors) + input_size
        for _ in range(mod_style_count):
            layers = []
            layers.append(nn.Linear(input_size, hidden_size, bias=True))
            layers.append(act_fn)
            for _ in range(layer_count-2):
                layers.append(nn.Linear(hidden_size, hidden_size, bias=True))
                layers.append(act_fn)
            layers.append(nn.Linear(hidden_size, latent_size, bias=True))
            self.networks.append(nn.Sequential(*layers))

    @property
    def id_latent(self):
        if self.w_opt.shape[1] == 1:
            return self.w_opt.expand(-1, self.style_count, -1)
        else:
            return self.w_opt

    def generate(self, x: torch.Tensor,
                 pitch: float,
                 yaw: float, return_x_only=False):

        alpha = (self.alpha[:, None] +
                 self.w_pitch[:, None] * pitch +
                 self.w_yaw[:, None] * yaw)
        alpha = self.get_constrained_alpha(alpha)
        w_plus = self.alpha_to_w(alpha)

        dynamic_latent, static_latent = w_plus.split(
            (self.mod_style_count, self.style_count-self.mod_style_count), -2)

        # x = torch.cat([x, alpha[None, :, 0, 0]], 1)
        delta = torch.stack([network(x) for network in self.networks], 1)

        if (self.style_count - self.mod_style_count) != 0:
            x = delta + dynamic_latent
            ws = torch.cat([x, static_latent], 1)
        else:
            ws = delta + dynamic_latent

        return ws[:, None], delta


class ExprAlphaV2(nn.Module):
    """Implement the module to record latent code in alpha-space (rotation code with
    an MLP).
    Additionally, it has an MLP to represent expression changes.
    """

    def __init__(
        self,
        stylegan: nn.Module,
        anchor_dir: Text,
        device: Text,
        n_samples: int = 10000,
        seed: int = 124,
        beta: float = 0.02,
        input_size: int = 113,
        hidden_size: int = 128,
        latent_size: int = 512,
        layer_count: int = 2,
        mod_style_count: int = 8,
        nonlinearity: Text = 'lrelu',
        use_mystyle: bool = False,
        ):
        super().__init__()
        # Compute mean latent code
        w_avg, w_std = compute_w_stats(stylegan, n_samples,
                                   seed=seed)

        # Compute mean latent code from the anchors
        if use_mystyle:
            paths = glob.glob(os.path.join(anchor_dir, '*.pt'))
        else:
            paths = glob.glob(os.path.join(anchor_dir, '*', '0.pt'))
        all_anchors = [torch.load(x).to(device) for x in paths]
        all_anchors = torch.cat(all_anchors, 0)
        self.register_buffer('all_anchors', all_anchors, persistent=True)
        alpha = torch.ones([1, len(self.all_anchors)])
        alpha = alpha / len(self.all_anchors)
        w_opt = self.alpha_to_w(alpha)
        self.beta = beta

        # self.w_opt = nn.Parameter(w_avg.clone().detach().requires_grad_(True))
        self.register_buffer('w_opt', w_opt.clone().detach().requires_grad_(False), persistent=True)
        self.register_buffer('alpha', alpha.clone().detach().requires_grad_(False), persistent=True)
        self.register_buffer('w_avg', w_avg, persistent=False)
        self.register_buffer('w_std', w_std, persistent=False)


        self.style_count = stylegan.num_ws
        self.mod_style_count = mod_style_count

        act_fn = None
        if nonlinearity == 'lrelu':
            act_fn = nn.LeakyReLU(inplace=True)
        elif nonlinearity == 'relu':
            act_fn = nn.ReLU(inplace=True)
        else:
            raise NotImplementedError

        self.expr_net = nn.ModuleList()
        # input_size = len(self.all_anchors) + input_size
        for _ in range(mod_style_count):
            layers = []
            layers.append(nn.Linear(input_size, hidden_size, bias=True))
            layers.append(act_fn)
            for _ in range(layer_count-2):
                layers.append(nn.Linear(hidden_size, hidden_size, bias=True))
                layers.append(act_fn)
            layers.append(nn.Linear(hidden_size, latent_size, bias=True))
            layers.append(nn.Tanh())
            self.expr_net.append(nn.Sequential(*layers))

        layers = []
        layers.append(nn.Linear(2, hidden_size, bias=True))
        layers.append(act_fn)
        for _ in range(layer_count-2):
            layers.append(nn.Linear(hidden_size, hidden_size, bias=True))
            layers.append(act_fn)
        layers.append(nn.Linear(hidden_size, len(self.all_anchors), bias=True))
        layers.append(nn.Tanh())
        self.alpha_net = nn.Sequential(*layers)

    @property
    def id_latent(self):
        if self.w_opt.shape[1] == 1:
            return self.w_opt.expand(-1, self.style_count, -1)
        else:
            return self.w_opt

    def alpha_to_w(self, alpha):
        ws = torch.einsum('bi...,i...->b...', alpha, self.all_anchors)
        return ws

    def get_constrained_alpha(self, alphas):
        if self.beta is not None:
            alphas = alphas + self.beta
            alphas = F.softplus(alphas, beta=100)
            alphas = alphas - self.beta

        alphas = alphas / torch.sum(alphas, dim=1, keepdim=True)

        return alphas

    def generate(self, x: torch.Tensor,
                 pitch: float,
                 yaw: float, return_alpha=False, **kwargs):

        feats = torch.cat([pitch, yaw], -1)
        rot_alpha = self.alpha_net(feats)

        alpha = (self.alpha + rot_alpha)
        alpha = self.get_constrained_alpha(alpha)
        w_plus = self.alpha_to_w(alpha)

        dynamic_latent, static_latent = w_plus.split(
            (self.mod_style_count, self.style_count-self.mod_style_count), -2)

        # x = torch.cat([x, alpha[None, :, 0, 0]], 1)
        delta = torch.stack([network(x) for network in self.expr_net], 1)

        if (self.style_count - self.mod_style_count) != 0:
            x = delta + dynamic_latent
            ws = torch.cat([x, static_latent], 1)
        else:
            ws = delta + dynamic_latent

        if return_alpha:
            return ws[:, None], delta, alpha
        return ws[:, None], delta

class ExprAlphaV3(nn.Module):
    """Implement the module to record latent code in alpha space (rotation code with
    an MLP).
    Additionally, it has an MLP to represent expression changes (in alpha plus space).
    """

    def __init__(
        self,
        stylegan: nn.Module,
        anchor_dir: Text,
        device: Text,
        n_samples: int = 10000,
        seed: int = 124,
        beta: float = 0.02,
        input_size: int = 113,
        hidden_size: int = 128,
        layer_count: int = 2,
        mod_style_count: int = 8,
        nonlinearity: Text = 'lrelu',
        ):
        super().__init__()
        # Compute mean latent code
        w_avg, w_std = compute_w_stats(stylegan, n_samples,
                                   seed=seed)

        # Compute mean latent code from the anchors
        paths = glob.glob(os.path.join(anchor_dir, '*.pt'))
        all_anchors = [torch.load(x).to(device) for x in paths]
        all_anchors = torch.cat(all_anchors, 0)
        self.register_buffer('all_anchors', all_anchors, persistent=True)
        alpha = torch.ones([1, len(self.all_anchors)])
        alpha = alpha / len(self.all_anchors)
        w_opt = self.alpha_to_w(alpha)
        self.beta = beta

        # self.w_opt = nn.Parameter(w_avg.clone().detach().requires_grad_(True))
        self.register_buffer('w_opt', w_opt.clone().detach().requires_grad_(False), persistent=True)
        self.register_buffer('alpha', alpha.clone().detach().requires_grad_(False), persistent=True)
        self.register_buffer('w_avg', w_avg, persistent=False)
        self.register_buffer('w_std', w_std, persistent=False)


        self.style_count = stylegan.num_ws
        self.mod_style_count = mod_style_count

        act_fn = None
        if nonlinearity == 'lrelu':
            act_fn = nn.LeakyReLU(inplace=True)
        elif nonlinearity == 'relu':
            act_fn = nn.ReLU(inplace=True)
        else:
            raise NotImplementedError

        self.expr_net = nn.ModuleList()
        # input_size = len(self.all_anchors) + input_size
        for _ in range(mod_style_count):
            layers = []
            layers.append(nn.Linear(input_size, hidden_size, bias=True))
            layers.append(act_fn)
            for _ in range(layer_count-2):
                layers.append(nn.Linear(hidden_size, hidden_size, bias=True))
                layers.append(act_fn)
            layers.append(nn.Linear(hidden_size, len(self.all_anchors), bias=True))
            layers.append(nn.Tanh())
            self.expr_net.append(nn.Sequential(*layers))

        layers = []
        layers.append(nn.Linear(2, hidden_size, bias=True))
        layers.append(act_fn)
        for _ in range(layer_count-2):
            layers.append(nn.Linear(hidden_size, hidden_size, bias=True))
            layers.append(act_fn)
        layers.append(nn.Linear(hidden_size, len(self.all_anchors), bias=True))
        layers.append(nn.Tanh())
        self.alpha_net = nn.Sequential(*layers)

    @property
    def id_latent(self):
        if self.w_opt.shape[1] == 1:
            return self.w_opt.expand(-1, self.style_count, -1)
        else:
            return self.w_opt

    def alpha_to_w(self, alpha):
        ws = torch.einsum('bi...,i...->b...', alpha, self.all_anchors)
        return ws

    def alpha_plus_to_w_plus(self, alpha):
        ws = torch.einsum('bji...,ij...->bj...', alpha, self.all_anchors)
        return ws

    def get_constrained_alpha(self, alphas):
        if self.beta is not None:
            alphas = alphas + self.beta
            alphas = F.softplus(alphas, beta=100)
            alphas = alphas - self.beta

        alphas = alphas / torch.sum(alphas, dim=-1, keepdim=True)

        return alphas

    def generate(self, x: torch.Tensor,
                 pitch: float,
                 yaw: float, return_x_only=False):

        feats = torch.cat([pitch, yaw], -1)
        rot_alpha = self.alpha_net(feats)

        alpha = (self.alpha + rot_alpha)
        delta = torch.stack([network(x) for network in self.expr_net], 1)

        alpha_plus = alpha[:, None].expand([-1, 18, -1])

        dynamic_latent, static_latent = alpha_plus.split(
            (self.mod_style_count, self.style_count-self.mod_style_count), -2)

        if (self.style_count - self.mod_style_count) != 0:
            x = delta + dynamic_latent
            alpha_plus = torch.cat([x, static_latent], 1)
        else:
            alpha_plus = delta + dynamic_latent

        alpha_plus = self.get_constrained_alpha(alpha_plus)
        w_plus = self.alpha_plus_to_w_plus(alpha_plus)

        return w_plus[:, None], delta
    
class ExprAlphaV4(nn.Module):
    """Implement the module to record latent code in alpha-space (rotation code with
    an MLP).
    Additionally, it has an MLP to represent expression changes and another one to record
    rotation residuals.
    """

    def __init__(
        self,
        stylegan: nn.Module,
        anchor_dir: Text,
        device: Text,
        n_samples: int = 10000,
        seed: int = 124,
        beta: float = 0.02,
        input_size: int = 113,
        hidden_size: int = 128,
        latent_size: int = 512,
        layer_count: int = 2,
        mod_style_count: int = 8,
        nonlinearity: Text = 'lrelu',
        ):
        super().__init__()
        # Compute mean latent code
        w_avg, w_std = compute_w_stats(stylegan, n_samples,
                                   seed=seed)

        # Compute mean latent code from the anchors
        paths = glob.glob(os.path.join(anchor_dir, '*.pt'))
        all_anchors = [torch.load(x).to(device) for x in paths]
        all_anchors = torch.cat(all_anchors, 0)
        self.register_buffer('all_anchors', all_anchors, persistent=True)
        alpha = torch.ones([1, len(self.all_anchors)])
        alpha = alpha / len(self.all_anchors)
        w_opt = self.alpha_to_w(alpha)
        self.beta = beta

        # self.w_opt = nn.Parameter(w_avg.clone().detach().requires_grad_(True))
        self.register_buffer('w_opt', w_opt.clone().detach().requires_grad_(False), persistent=True)
        self.register_buffer('alpha', alpha.clone().detach().requires_grad_(False), persistent=True)
        self.register_buffer('w_avg', w_avg, persistent=False)
        self.register_buffer('w_std', w_std, persistent=False)


        self.style_count = stylegan.num_ws
        self.mod_style_count = mod_style_count

        act_fn = None
        if nonlinearity == 'lrelu':
            act_fn = nn.LeakyReLU(inplace=True)
        elif nonlinearity == 'relu':
            act_fn = nn.ReLU(inplace=True)
        else:
            raise NotImplementedError

        self.expr_net = nn.ModuleList()
        # input_size = len(self.all_anchors) + input_size
        for _ in range(mod_style_count):
            layers = []
            layers.append(nn.Linear(input_size, hidden_size, bias=True))
            layers.append(act_fn)
            for _ in range(layer_count-2):
                layers.append(nn.Linear(hidden_size, hidden_size, bias=True))
                layers.append(act_fn)
            layers.append(nn.Linear(hidden_size, latent_size, bias=True))
            layers.append(nn.Tanh())
            self.expr_net.append(nn.Sequential(*layers))

        self.res_net = nn.ModuleList()
        # input_size = len(self.all_anchors) + input_size
        for _ in range(mod_style_count):
            layers = []
            layers.append(nn.Linear(2, hidden_size, bias=True))
            layers.append(act_fn)
            for _ in range(layer_count-2):
                layers.append(nn.Linear(hidden_size, hidden_size, bias=True))
                layers.append(act_fn)
            layers.append(nn.Linear(hidden_size, latent_size, bias=True))
            layers.append(nn.Tanh())
            self.res_net.append(nn.Sequential(*layers))

        layers = []
        layers.append(nn.Linear(2, hidden_size, bias=True))
        layers.append(act_fn)
        for _ in range(layer_count-2):
            layers.append(nn.Linear(hidden_size, hidden_size, bias=True))
            layers.append(act_fn)
        layers.append(nn.Linear(hidden_size, len(self.all_anchors), bias=True))
        layers.append(nn.Tanh())
        self.alpha_net = nn.Sequential(*layers)

    @property
    def id_latent(self):
        if self.w_opt.shape[1] == 1:
            return self.w_opt.expand(-1, self.style_count, -1)
        else:
            return self.w_opt

    def alpha_to_w(self, alpha):
        ws = torch.einsum('bi...,i...->b...', alpha, self.all_anchors)
        return ws

    def get_constrained_alpha(self, alphas):
        if self.beta is not None:
            alphas = alphas + self.beta
            alphas = F.softplus(alphas, beta=100)
            alphas = alphas - self.beta

        alphas = alphas / torch.sum(alphas, dim=1, keepdim=True)

        return alphas

    def generate(self, x: torch.Tensor,
                 pitch: float,
                 yaw: float, return_x_only=False):

        feats = torch.cat([pitch, yaw], -1)
        rot_alpha = self.alpha_net(feats)
        rot_residual = torch.stack([network(feats) for network in self.res_net], 1)

        alpha = (self.alpha + rot_alpha)
        alpha = self.get_constrained_alpha(alpha)
        w_plus = self.alpha_to_w(alpha)

        dynamic_latent, static_latent = w_plus.split(
            (self.mod_style_count, self.style_count-self.mod_style_count), -2)

        delta = torch.stack([network(x) for network in self.expr_net], 1) + \
                rot_residual

        if (self.style_count - self.mod_style_count) != 0:
            x = delta + dynamic_latent
            ws = torch.cat([x, static_latent], 1)
        else:
            ws = delta + dynamic_latent

        return ws[:, None], delta