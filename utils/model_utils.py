import numpy as np
import pickle
import torch
import torch.nn as nn

def set_requires_grad(requires_grad, *models):
    for model in models:
        if isinstance(model, nn.Module):
            for param in model.parameters():
                param.requires_grad = requires_grad
        elif isinstance(model, (nn.Parameter, torch.Tensor)):
            model.requires_grad = requires_grad
        else:
            assert False, 'unknown type %r' % type(model)

def load_generator(path, device, training=False):
    with open(path, 'rb') as f:
        generator = pickle.load(f)['G_ema'].to(device)
        generator = generator.float()
    if not training:
        generator.eval()
        set_requires_grad(False, generator)
    return generator

def compute_w_stats(generator: nn.Module,
                    n_samples: int = 1000,
                    seed: int = 123,
                    device: str = None):
    """Compute the mean and standard deviation of the W code.

    Args:
        generator: StyleGAN2 generator.
        n_samples: How many samples to use.
        seed: Random seed for the RNG.
    Returns:
        A tuple of the mean and standard deviation of the W code.
        (mean, std)
    """
    z_samples = np.random.RandomState(seed=seed).randn(n_samples,
                                                       generator.z_dim)
    w_samples = generator.mapping(
        torch.tensor(z_samples, device=device), None)
    w_std, w_avg = torch.std_mean(w_samples[:, :1, :], dim=0,
                                  unbiased=False, keepdim=True)
    return w_avg, w_std
