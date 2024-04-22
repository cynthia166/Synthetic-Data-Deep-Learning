#from __future__ import division
#f#rom __future__ import unicode_literal
from __future__ import unicode_literals
import os
import logging
import torch
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import sys
sys.path.append('.')
#from model.linear_model import LinearModel
#from utils.util import set_seeds, make_dir, save_checkpoint, sample_random_batch, plot_dim_dist
#from model.ema import ExponentialMovingAverage
#from score_losses import EDMLoss, VPSDELoss, VESDELoss#, VLoss
#from denoiser import EDMDenoiser, VPSDEDenoiser, VESDEDenoiser, NaiveDenoiser#, VDenoiser
#from samplers import ablation_sampler#, ddim_sampler, edm_sampler

import importlib
opacus = importlib.import_module('src.opacus')
import pandas as pd
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP

from transformers import get_cosine_schedule_with_warmup


import math
import torch
import torch.nn as nn
from einops import rearrange

import torch
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.distributed as dist
# import PIL
# from torchvision.utils import make_grid
from scipy import linalg
from scipy.stats import pearsonr
from pathlib import Path

# from dataset_tool import is_image_ext

# Modified from https://raw.githubusercontent.com/fadel/pytorch_ema/master/torch_ema/ema.py



import torch
import torch
import torch.nn
import numpy as np
import math

#from utils.util import add_dimensions

import torch
import torch.nn
import numpy as np
import math

#from utils.util import add_dimensions


import math
import numpy as np
import torch
import torch.nn as nn

import torch
import numpy as np

import torch
import numpy as np


def guidance_wrapper(denoiser, guid_scale):
    def guidance_denoiser(x, t, y):
        if guid_scale > 0:
            no_class_label = denoiser.module.model.label_dim * torch.ones_like(y, device=x.device)
            return (1. + guid_scale) * denoiser(x, t, y) - guid_scale * denoiser(x, t, no_class_label)
        else:
            return denoiser(x, t, y)

    return guidance_denoiser


# https://github.com/NVlabs/edm/blob/main/generate.py
def ablation_sampler(
    latents, class_labels, net, randn_like=torch.randn_like, guid_scale=None, stochastic=False,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    solver='heun', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    if guid_scale is not None:
        net = guidance_wrapper(net, guid_scale)

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    try:
        sigma_min = max(sigma_min, net.module.sigma_min)
        sigma_max = min(sigma_max, net.module.sigma_max)
    except:
        pass

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(torch.as_tensor(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        ###
        if not stochastic:
            gamma = 0
        ###
        t_hat = sigma_inv(torch.as_tensor(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
        d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == 'euler' or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == 'heun'
            denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels).to(torch.float64)
            d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

    return x_next

def guidance_wrapper(denoiser, guid_scale):
    def guidance_denoiser(x, t, y):
        if guid_scale > 0:
            no_class_label = denoiser.module.model.label_dim * torch.ones_like(y, device=x.device)
            return (1. + guid_scale) * denoiser(x, t, y) - guid_scale * denoiser(x, t, no_class_label)
        else:
            return denoiser(x, t, y)

    return guidance_denoiser


# https://github.com/NVlabs/edm/blob/main/generate.py
def ablation_sampler(
    latents, class_labels, net, randn_like=torch.randn_like, guid_scale=None, stochastic=False,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    solver='heun', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    if guid_scale is not None:
        net = guidance_wrapper(net, guid_scale)

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    try:
        sigma_min = max(sigma_min, net.module.sigma_min)
        sigma_max = min(sigma_max, net.module.sigma_max)
    except:
        pass

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(torch.as_tensor(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        ###
        if not stochastic:
            gamma = 0
        ###
        t_hat = sigma_inv(torch.as_tensor(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
        d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == 'euler' or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == 'heun'
            denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels).to(torch.float64)
            d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

    return x_next


class NaiveDenoiser(nn.Module):
    def __init__(self,
                model,
                ):

        super().__init__()
        self.model = model

    def forward(self, x, sigma, y=None):
        x = x.to(torch.float32)
        return self.model(x, sigma.reshape(-1), y)


class EDMDenoiser(nn.Module):
    def __init__(self,
                model,
                sigma_min,
                sigma_max,
                sigma_data=math.sqrt(1. / 3)
                ):

        super().__init__()

        self.sigma_data = sigma_data
        self.model = model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def forward(self, x, sigma, y=None):
        x = x.to(torch.float32)
        c_skip = self.sigma_data ** 2. / \
            (sigma ** 2. + self.sigma_data ** 2.)
        c_out = sigma * self.sigma_data / \
            torch.sqrt(self.sigma_data ** 2. + sigma ** 2.)
        c_in = 1. / torch.sqrt(self.sigma_data ** 2. + sigma ** 2.)
        c_noise = .25 * torch.log(sigma)

        out = self.model(c_in * x, c_noise.reshape(-1), y)

        x_denoised = c_skip * x + c_out * out
        return x_denoised


class VDenoiser(nn.Module):
    def __init__(
                self,
                model
                ):

        super().__init__()
        self.model = model

    def _sigma_inv(self, sigma):
        return 2. * torch.arccos(1. / (1. + sigma ** 2.).sqrt()) / np.pi

    def forward(self, x, sigma, y=None):
        x = x.to(torch.float32)
        c_skip = 1. / (sigma ** 2. + 1.)
        c_out = sigma / torch.sqrt(1. + sigma ** 2.)
        c_in = 1. / torch.sqrt(1. + sigma ** 2.)
        c_noise = self._sigma_inv(sigma)

        out = self.model(c_in * x, c_noise.reshape(-1), y)
        x_denoised = c_skip * x + c_out * out
        return x_denoised
    

class VESDEDenoiser(nn.Module):
    def __init__(self,
                sigma_min,
                sigma_max,
                model,
                ):

        super().__init__()

        self.model = model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def forward(self, x, sigma, y=None):
        
        x = x.to(torch.float32)

        c_skip = 1. 
        ### Essential adjustment for mimic data
        # c_skip = 0.11 ** 2. / \
        #     (sigma ** 2. + 0.11 ** 2.)
        
        c_out = sigma
        c_in = 1.
        c_noise = torch.log(sigma / 2.)

        out = self.model(c_in * x, c_noise.reshape(-1), y)
        x_denoised = c_skip * x + c_out * out
        return x_denoised

    

class VPSDEDenoiser(nn.Module):
    def __init__(
                self,
                beta_min,
                beta_d,
                M,
                eps_t,
                model
                ):

        super().__init__()

        self.model = model
        self.M = M
        self.beta_min = beta_min
        self.beta_d = beta_d
        ### https://github.com/NVlabs/edm/blob/main/training/networks.py
        self.sigma_min = float(self.sigma(eps_t))
        self.sigma_max = float(self.sigma(1))

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()
    
    def _sigma_inv(self, sigma):
        sigma = torch.as_tensor(sigma)
        return ((self.beta_min ** 2 + 2 * self.beta_d * (1 + sigma ** 2).log()).sqrt() - self.beta_min) / self.beta_d

    def forward(self, x, sigma, y=None):

        x = x.to(torch.float32)
        
        c_skip = 1.
        ### Essential adjustment for mimic data
        # c_skip = 0.13 ** 2. / \
        #     (sigma ** 2. + 0.13 ** 2.)
        
        c_out = -sigma
        c_in = 1. / torch.sqrt(sigma ** 2. + 1.)
        c_noise = (self.M-1) * self._sigma_inv(sigma)

        out = self.model(c_in * x, c_noise.reshape(-1), y)
        x_denoised = c_skip * x + c_out * out
        return x_denoised
    
def dropout_label_for_cfg_training(y, n_noise_samples, n_classes, p, device):
    if y is not None:
        if n_classes is None:
            raise ValueError
        else:
            with torch.no_grad():
                boolean_ = torch.bernoulli(
                    p * torch.ones_like(y, device=device)).bool()
                no_class_label = n_classes * torch.ones_like(y, device=device)
                y = torch.where(boolean_, no_class_label, y)
                y = y.repeat_interleave(n_noise_samples)
                return y
    else:
        return None


class VPSDELoss:
    def __init__(self, beta_min, beta_d, eps_t, n_noise_samples=1, label_unconditioning_prob=.1, n_classes=None, **kwargs):
        self.beta_min = beta_min
        self.beta_d = beta_d
        self.eps_t = eps_t
        self.n_noise_samples = n_noise_samples
        self.label_unconditioning_prob = label_unconditioning_prob
        self.n_classes = n_classes

    def _sigma(self, t):
        return ((.5 * self.beta_d * t ** 2. + self.beta_min * t).exp() - 1.).sqrt()

    def get_loss(self, model, x, y):

        y = dropout_label_for_cfg_training(
            y, self.n_noise_samples, self.n_classes, self.label_unconditioning_prob, x.device)

        ### https://github.com/NVlabs/edm/blob/main/training/loss.py
        t = 1 + torch.rand((x.shape[0], self.n_noise_samples),device=x.device) * (self.eps_t - 1)
        sigma = self._sigma(t) 

        sigma = add_dimensions(sigma, len(x.shape) - 1)
        x_repeated = x.unsqueeze(1).repeat_interleave(
            self.n_noise_samples, dim=1)
        x_noisy = x_repeated + sigma * \
            torch.randn_like(x_repeated, device=x.device)

        w = 1. / sigma ** 2.

        pred = model(x_noisy.reshape(-1, *x.shape[1:]), sigma.reshape(-1, *sigma.shape[2:]), y).reshape(
            x.shape[0], self.n_noise_samples, *x.shape[1:])
        loss = w * (pred - x_repeated) ** 2.

        loss = torch.mean(loss.reshape(loss.shape[0], -1), dim=-1)
        return loss


class VESDELoss:
    def __init__(self, sigma_min, sigma_max, n_noise_samples=1, label_unconditioning_prob=.1, n_classes=None, **kwargs):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.n_noise_samples = n_noise_samples
        self.label_unconditioning_prob = label_unconditioning_prob
        self.n_classes = n_classes

    def get_loss(self, model, x, y):
        y = dropout_label_for_cfg_training(
            y, self.n_noise_samples, self.n_classes, self.label_unconditioning_prob, x.device)

        log_sigma = (np.log(self.sigma_max) - np.log(self.sigma_min)) * torch.rand(
            (x.shape[0], self.n_noise_samples), device=x.device) + np.log(self.sigma_min)
        sigma = log_sigma.exp()

        sigma = add_dimensions(sigma, len(x.shape) - 1)
        x_repeated = x.unsqueeze(1).repeat_interleave(
            self.n_noise_samples, dim=1)
        x_noisy = x_repeated + sigma * \
            torch.randn_like(x_repeated, device=x.device)

        w = 1. / sigma ** 2.

        pred = model(x_noisy.reshape(-1, *x.shape[1:]), sigma.reshape(-1, *sigma.shape[2:]), y).reshape(
            x.shape[0], self.n_noise_samples, *x.shape[1:])
        
        loss = w * (pred - x_repeated) ** 2.
        loss = torch.mean(loss.reshape(loss.shape[0], -1), dim=-1)
        return loss


class VLoss:
    def __init__(self, logsnr_min, logsnr_max, n_noise_samples=1, label_unconditioning_prob=.1, n_classes=None, **kwargs):
        self.logsnr_min = logsnr_min
        self.logsnr_max = logsnr_max
        self.eps_min = self._t_given_logsnr(logsnr_max)
        self.eps_max = self._t_given_logsnr(logsnr_min)
        self.n_noise_samples = n_noise_samples
        self.label_unconditioning_prob = label_unconditioning_prob
        self.n_classes = n_classes

    def _t_given_logsnr(self, logsnr):
        return 2. * np.arccos(1. / np.sqrt(1. + np.exp(-logsnr))) / np.pi

    def _sigma(self, t):
        return (torch.cos(np.pi * t / 2.) ** (-2.) - 1.).sqrt()

    def get_loss(self, model, x, y):
        y = dropout_label_for_cfg_training(
            y, self.n_noise_samples, self.n_classes, self.label_unconditioning_prob, x.device)

        t = (self.eps_max - self.eps_min) * \
            torch.rand((x.shape[0], self.n_noise_samples),
                       device=x.device) + self.eps_min
        sigma = self._sigma(t)

        sigma = add_dimensions(sigma, len(x.shape) - 1)
        x_repeated = x.unsqueeze(1).repeat_interleave(
            self.n_noise_samples, dim=1)
        x_noisy = x_repeated + sigma * \
            torch.randn_like(x_repeated, device=x.device)

        w = (sigma ** 2. + 1.) / sigma ** 2.

        pred = model(x_noisy.reshape(-1, *x.shape[1:]), sigma.reshape(-1, *sigma.shape[2:]), y).reshape(
            x.shape[0], self.n_noise_samples, *x.shape[1:])
        loss = w * (pred - x_repeated) ** 2.
        loss = torch.mean(loss.reshape(loss.shape[0], -1), dim=-1)
        return loss


class EDMLoss:
    def __init__(self, p_mean, p_std, sigma_data=math.sqrt(1. / 3), n_noise_samples=1, label_unconditioning_prob=.1, n_classes=None, **kwargs):
        self.p_mean = p_mean
        self.p_std = p_std
        self.sigma_data = sigma_data
        self.n_noise_samples = n_noise_samples
        self.label_unconditioning_prob = label_unconditioning_prob
        self.n_classes = n_classes

    def get_loss(self, model, x, y):
        y = dropout_label_for_cfg_training(
            y, self.n_noise_samples, self.n_classes, self.label_unconditioning_prob, x.device)

        log_sigma = self.p_mean + self.p_std * \
            torch.randn(
                (x.shape[0], self.n_noise_samples), device=x.device)

        sigma = log_sigma.exp()

        sigma = add_dimensions(sigma, len(x.shape) - 1)
        x_repeated = x.unsqueeze(1).repeat_interleave(
            self.n_noise_samples, dim=1)
        x_noisy = x_repeated + sigma * \
            torch.randn_like(x_repeated, device=x.device)

        w = (sigma ** 2. + self.sigma_data ** 2.) / \
            (sigma * self.sigma_data) ** 2.

        pred = model(x_noisy.reshape(-1, *x.shape[1:]), sigma.reshape(-1, *sigma.shape[2:]), y).reshape(
            x.shape[0], self.n_noise_samples, *x.shape[1:])
        loss = w * (pred - x_repeated) ** 2.
        loss = torch.mean(loss.reshape(loss.shape[0], -1), dim=-1)
        return loss

def dropout_label_for_cfg_training(y, n_noise_samples, n_classes, p, device):
    if y is not None:
        if n_classes is None:
            raise ValueError
        else:
            with torch.no_grad():
                boolean_ = torch.bernoulli(
                    p * torch.ones_like(y, device=device)).bool()
                no_class_label = n_classes * torch.ones_like(y, device=device)
                y = torch.where(boolean_, no_class_label, y)
                y = y.repeat_interleave(n_noise_samples)
                return y
    else:
        return None


class VPSDELoss:
    def __init__(self, beta_min, beta_d, eps_t, n_noise_samples=1, label_unconditioning_prob=.1, n_classes=None, **kwargs):
        self.beta_min = beta_min
        self.beta_d = beta_d
        self.eps_t = eps_t
        self.n_noise_samples = n_noise_samples
        self.label_unconditioning_prob = label_unconditioning_prob
        self.n_classes = n_classes

    def _sigma(self, t):
        return ((.5 * self.beta_d * t ** 2. + self.beta_min * t).exp() - 1.).sqrt()

    def get_loss(self, model, x, y):

        y = dropout_label_for_cfg_training(
            y, self.n_noise_samples, self.n_classes, self.label_unconditioning_prob, x.device)

        ### https://github.com/NVlabs/edm/blob/main/training/loss.py
        t = 1 + torch.rand((x.shape[0], self.n_noise_samples),device=x.device) * (self.eps_t - 1)
        sigma = self._sigma(t) 

        sigma = add_dimensions(sigma, len(x.shape) - 1)
        x_repeated = x.unsqueeze(1).repeat_interleave(
            self.n_noise_samples, dim=1)
        x_noisy = x_repeated + sigma * \
            torch.randn_like(x_repeated, device=x.device)

        w = 1. / sigma ** 2.

        pred = model(x_noisy.reshape(-1, *x.shape[1:]), sigma.reshape(-1, *sigma.shape[2:]), y).reshape(
            x.shape[0], self.n_noise_samples, *x.shape[1:])
        loss = w * (pred - x_repeated) ** 2.

        loss = torch.mean(loss.reshape(loss.shape[0], -1), dim=-1)
        return loss


class VESDELoss:
    def __init__(self, sigma_min, sigma_max, n_noise_samples=1, label_unconditioning_prob=.1, n_classes=None, **kwargs):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.n_noise_samples = n_noise_samples
        self.label_unconditioning_prob = label_unconditioning_prob
        self.n_classes = n_classes

    def get_loss(self, model, x, y):
        y = dropout_label_for_cfg_training(
            y, self.n_noise_samples, self.n_classes, self.label_unconditioning_prob, x.device)

        log_sigma = (np.log(self.sigma_max) - np.log(self.sigma_min)) * torch.rand(
            (x.shape[0], self.n_noise_samples), device=x.device) + np.log(self.sigma_min)
        sigma = log_sigma.exp()

        sigma = add_dimensions(sigma, len(x.shape) - 1)
        x_repeated = x.unsqueeze(1).repeat_interleave(
            self.n_noise_samples, dim=1)
        x_noisy = x_repeated + sigma * \
            torch.randn_like(x_repeated, device=x.device)

        w = 1. / sigma ** 2.

        pred = model(x_noisy.reshape(-1, *x.shape[1:]), sigma.reshape(-1, *sigma.shape[2:]), y).reshape(
            x.shape[0], self.n_noise_samples, *x.shape[1:])
        
        loss = w * (pred - x_repeated) ** 2.
        loss = torch.mean(loss.reshape(loss.shape[0], -1), dim=-1)
        return loss


class VLoss:
    def __init__(self, logsnr_min, logsnr_max, n_noise_samples=1, label_unconditioning_prob=.1, n_classes=None, **kwargs):
        self.logsnr_min = logsnr_min
        self.logsnr_max = logsnr_max
        self.eps_min = self._t_given_logsnr(logsnr_max)
        self.eps_max = self._t_given_logsnr(logsnr_min)
        self.n_noise_samples = n_noise_samples
        self.label_unconditioning_prob = label_unconditioning_prob
        self.n_classes = n_classes

    def _t_given_logsnr(self, logsnr):
        return 2. * np.arccos(1. / np.sqrt(1. + np.exp(-logsnr))) / np.pi

    def _sigma(self, t):
        return (torch.cos(np.pi * t / 2.) ** (-2.) - 1.).sqrt()

    def get_loss(self, model, x, y):
        y = dropout_label_for_cfg_training(
            y, self.n_noise_samples, self.n_classes, self.label_unconditioning_prob, x.device)

        t = (self.eps_max - self.eps_min) * \
            torch.rand((x.shape[0], self.n_noise_samples),
                       device=x.device) + self.eps_min
        sigma = self._sigma(t)

        sigma = add_dimensions(sigma, len(x.shape) - 1)
        x_repeated = x.unsqueeze(1).repeat_interleave(
            self.n_noise_samples, dim=1)
        x_noisy = x_repeated + sigma * \
            torch.randn_like(x_repeated, device=x.device)

        w = (sigma ** 2. + 1.) / sigma ** 2.

        pred = model(x_noisy.reshape(-1, *x.shape[1:]), sigma.reshape(-1, *sigma.shape[2:]), y).reshape(
            x.shape[0], self.n_noise_samples, *x.shape[1:])
        loss = w * (pred - x_repeated) ** 2.
        loss = torch.mean(loss.reshape(loss.shape[0], -1), dim=-1)
        return loss


class EDMLoss:
    def __init__(self, p_mean, p_std, sigma_data=math.sqrt(1. / 3), n_noise_samples=1, label_unconditioning_prob=.1, n_classes=None, **kwargs):
        self.p_mean = p_mean
        self.p_std = p_std
        self.sigma_data = sigma_data
        self.n_noise_samples = n_noise_samples
        self.label_unconditioning_prob = label_unconditioning_prob
        self.n_classes = n_classes

    def get_loss(self, model, x, y):
        y = dropout_label_for_cfg_training(
            y, self.n_noise_samples, self.n_classes, self.label_unconditioning_prob, x.device)

        log_sigma = self.p_mean + self.p_std * \
            torch.randn(
                (x.shape[0], self.n_noise_samples), device=x.device)

        sigma = log_sigma.exp()

        sigma = add_dimensions(sigma, len(x.shape) - 1)
        x_repeated = x.unsqueeze(1).repeat_interleave(
            self.n_noise_samples, dim=1)
        x_noisy = x_repeated + sigma * \
            torch.randn_like(x_repeated, device=x.device)

        w = (sigma ** 2. + self.sigma_data ** 2.) / \
            (sigma * self.sigma_data) ** 2.

        pred = model(x_noisy.reshape(-1, *x.shape[1:]), sigma.reshape(-1, *sigma.shape[2:]), y).reshape(
            x.shape[0], self.n_noise_samples, *x.shape[1:])
        loss = w * (pred - x_repeated) ** 2.
        loss = torch.mean(loss.reshape(loss.shape[0], -1), dim=-1)
        return loss

# Partially based on: https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/moving_averages.py
class ExponentialMovingAverage:
  """
  Maintains (exponential) moving average of a set of parameters.
  """

  def __init__(self, parameters, decay, use_num_updates=True):
    """
    Args:
      parameters: Iterable of `torch.nn.Parameter`; usually the result of
        `model.parameters()`.
      decay: The exponential decay.
      use_num_updates: Whether to use number of updates when computing
        averages.
    """
    if decay < 0.0 or decay > 1.0:
      raise ValueError('Decay must be between 0 and 1')
    self.decay = decay
    self.num_updates = 0 if use_num_updates else None
    self.shadow_params = [p.clone().detach()
                          for p in parameters if p.requires_grad]
    self.collected_params = []

  def update(self, parameters):
    """
    Update currently maintained parameters.

    Call this every time the parameters are updated, such as the result of
    the `optimizer.step()` call.

    Args:
      parameters: Iterable of `torch.nn.Parameter`; usually the same set of
        parameters used to initialize this object.
    """
    decay = self.decay
    if self.num_updates is not None:
      self.num_updates += 1
      decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
    one_minus_decay = 1.0 - decay
    with torch.no_grad():
      parameters = [p for p in parameters if p.requires_grad]
      for s_param, param in zip(self.shadow_params, parameters):
        s_param.sub_(one_minus_decay * (s_param - param))

  def copy_to(self, parameters):
    """
    Copy current parameters into given collection of parameters.

    Args:
      parameters: Iterable of `torch.nn.Parameter`; the parameters to be
        updated with the stored moving averages.
    """
    parameters = [p for p in parameters if p.requires_grad]
    for s_param, param in zip(self.shadow_params, parameters):
      if param.requires_grad:
        param.data.copy_(s_param.data)

  def store(self, parameters):
    """
    Save the current parameters for restoring later.

    Args:
      parameters: Iterable of `torch.nn.Parameter`; the parameters to be
        temporarily stored.
    """
    self.collected_params = [param.clone() for param in parameters]

  def restore(self, parameters):
    """
    Restore the parameters stored with the `store` method.
    Useful to validate the model with EMA parameters without affecting the
    original optimization process. Store the parameters before the
    `copy_to` method. After validation (or model saving), use this to
    restore the former parameters.

    Args:
      parameters: Iterable of `torch.nn.Parameter`; the parameters to be
        updated with the stored parameters.
    """
    for c_param, param in zip(self.collected_params, parameters):
      param.data.copy_(c_param.data)

  def state_dict(self):
    return dict(decay=self.decay, num_updates=self.num_updates,
                shadow_params=self.shadow_params)

  def load_state_dict(self, state_dict):
    self.decay = state_dict['decay']
    self.num_updates = state_dict['num_updates']
    self.shadow_params = state_dict['shadow_params']
    
def average_tensor(t):
    size = float(dist.get_world_size())
    dist.all_reduce(t.data, op=dist.ReduceOp.SUM)
    t.data /= size


def set_seeds(rank, seed):
    random.seed(rank + seed)
    torch.manual_seed(rank + seed)
    np.random.seed(rank + seed)
    torch.cuda.manual_seed(rank + seed)
    torch.cuda.manual_seed_all(rank + seed)
    torch.backends.cudnn.benchmark = True


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    # else:
    #     raise ValueError('Directory already exists.')


def add_dimensions(x, n_additional_dims):
    for _ in range(n_additional_dims):
        x = x.unsqueeze(-1)

    return x


def add_dimensions(x, n_additional_dims):
    for _ in range(n_additional_dims):
        x = x.unsqueeze(-1)

    return x

def save_checkpoint(ckpt_path, state):
    saved_state = {'model': state['model'].state_dict(),
                   'ema': state['ema'].state_dict(),
                   'optimizer': state['optimizer'].state_dict(),
                   'step': state['step']}
    torch.save(saved_state, ckpt_path)


def sample_random_batch(EHR_task, sampling_shape, sampler, path, device, n_classes=None, name='sample'):
    # make_dir(path)

    x = torch.randn(sampling_shape, device=device)
    if n_classes is not None:
        y = torch.randint(n_classes, size=(
            sampling_shape[0],), dtype=torch.int32, device=device)
    else:
        y = None
    x = sampler(x, y).cpu()

    # x[np.isnan(x)] = 0

    if EHR_task == 'binary':
        x = np.rint(np.clip(x, 0, 1))
    elif EHR_task == 'continuous':
        x = np.clip(x, 0, 1)

    np.save(os.path.join(path, name), x)


# ------------------------------------------------------------------------------------
def plot_dim_dist(train_data, syn_data, save_dir):

    train_data_mean = np.mean(train_data, axis = 0)
    temp_data_mean = np.mean(syn_data, axis = 0)

    corr = pearsonr(temp_data_mean, train_data_mean)
    nzc = sum(temp_data_mean[i] > 0 for i in range(temp_data_mean.shape[0]))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    slope, intercept = np.polyfit(train_data_mean, temp_data_mean, 1)
    fitted_values = [slope * i + intercept for i in train_data_mean]
    identity_values = [1 * i + 0 for i in train_data_mean]

    ax.plot(train_data_mean, fitted_values, 'b', alpha=0.5)
    ax.plot(train_data_mean, identity_values, 'r', alpha=0.5)
    ax.scatter(train_data_mean, temp_data_mean, alpha=0.3)
    ax.set_title('corr: %.4f, none-zero columns: %d, slope: %.4f'%(corr[0], nzc, slope))
    ax.set_xlabel('Feature prevalence of real data')
    ax.set_ylabel('Feature prevalence of synthetic data')

    fig.savefig(save_dir + '/{}.png'.format('Dimension-Wise Distribution'))
    plt.close(fig)

    return corr[0], nzc

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))


class Block(nn.Module):
    def __init__(self, dim_in, dim_out, *, time_emb_dim=None):
        super().__init__()

        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, dim_in),
            )
        
        self.out_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim_in, dim_out),
        )

    def forward(self, x, time_emb=None):
        
        if time_emb is not None:
            t_emb = self.time_mlp(time_emb)
            h = x + t_emb  
        else:
            h = x
        out = self.out_proj(h)
        return out  


class LinearModel(nn.Module):
    def __init__(
            self, *,
            z_dim=666, 
            time_dim=384,
            unit_dims=[1024, 384, 384, 384, 1024],

            random_fourier_features=False,
            learned_sinusoidal_dim=32,

            use_cfg=False,
            num_classes=2,
            class_dim=128,
            ):
        super().__init__()
        
        num_linears = len(unit_dims)

        if random_fourier_features:
            self.time_embedding = nn.Sequential(
                RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, is_random=True),
                nn.Linear(learned_sinusoidal_dim+1, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            self.time_embedding = nn.Sequential(
                    SinusoidalPositionEmbeddings(z_dim),
                    nn.Linear(z_dim, time_dim),
                    nn.SiLU(),
                    nn.Linear(time_dim, time_dim),
                )

        self.block_in = Block(dim_in=z_dim, dim_out=unit_dims[0], time_emb_dim=time_dim)
        self.block_mid = nn.ModuleList()
        for i in range(num_linears-1):
            self.block_mid.append(Block(dim_in=unit_dims[i], dim_out=unit_dims[i+1], time_emb_dim=time_dim))
        self.block_out = Block(dim_in=unit_dims[-1], dim_out=z_dim, time_emb_dim=time_dim)

        ### Classifier-free 
        self.label_dim = num_classes
        self.use_cfg = use_cfg
        if use_cfg:
            self.class_emb = nn.Embedding(self.label_dim if not use_cfg else self.label_dim + 1, class_dim)
            self.class_mlp = nn.Sequential(
                nn.Linear(class_dim, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim)
            )    

    def forward(self, x, time_steps, labels=None):
        
        time_steps = time_steps.float()
        t_emb = self.time_embedding(time_steps)
        if self.use_cfg:
            class_emb = self.class_mlp(self.class_emb(labels))
            t_emb += class_emb 

        x = self.block_in(x, t_emb)

        num_mid_blocks = len(self.block_mid)
        if num_mid_blocks > 0:
            for block in self.block_mid:
                x = block(x, t_emb)

        x = self.block_out(x, t_emb)
        return x


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered
    
def train_dpdm_base(config, workdir, mode):
    
    set_seeds(config.setup.global_rank, config.train.seed)
    torch.cuda.device(config.setup.local_rank)
    config.setup.device = 'cuda:%d' % config.setup.local_rank

    sample_dir = os.path.join(workdir, 'samples')
    checkpoint_dir = os.path.join(workdir, 'checkpoints')

    if config.setup.global_rank == 0:
        if mode == 'train':
            make_dir(sample_dir)
            make_dir(checkpoint_dir)
    dist.barrier()

    if config.model.denoiser_name == 'edm':
        if config.model.denoiser_network == 'song':
            model = EDMDenoiser(
                model=LinearModel(**config.model.network).to(config.setup.device), **config.model.params)
            # model = NaiveDenoiser(
            #     model=LinearModel(**config.model.network).to(config.setup.device))          
        else:
            raise NotImplementedError
    elif config.model.denoiser_name == 'vpsde':
        if config.model.denoiser_network == 'song':
            model = VPSDEDenoiser(
                model=LinearModel(**config.model.network).to(config.setup.device), **config.model.params)
        else:
            raise NotImplementedError
    elif config.model.denoiser_name == 'vesde':
        if config.model.denoiser_network == 'song':
            model = VESDEDenoiser(
                model=LinearModel(**config.model.network).to(config.setup.device), **config.model.params)
        else:
            raise NotImplementedError
    elif config.model.denoiser_name == 'naive':
            model = NaiveDenoiser(
                model=LinearModel(**config.model.network).to(config.setup.device))
    # elif config.model.denoiser_name == 'v':
    #     if config.model.denoiser_network == 'song':
    #         model = VDenoiser(
    #             model=LinearModel(**config.model.network).to(config.setup.device), **config.model.params)
    #     else:
    #         raise NotImplementedError
    else:
        raise NotImplementedError


    if config.dp.do:
        model = DPDDP(model)
    else:
        model = DistributedDataParallel(model.to(config.setup.device), device_ids=[config.setup.device])

    ema = ExponentialMovingAverage(
        model.parameters(), decay=config.model.ema_rate)

    if config.optim.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), **config.optim.params)
    elif config.optim.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), **config.optim.params)
    else:
        raise NotImplementedError
    raw_data = np.load(config.data.path)
    print(raw_data.shape) 
    import pickle
    import gzip
    '''with gzip.open('train_sp' + '/train_splitDATASET_NAME_prepro' + 'train_data_features.pkl','rb') as f:
         raw_data = pickle.load(f)
         
    raw_data = raw_data[:1000]
    raw_data = np.transpose(raw_data, (0, 2, 1))
    N, F, T = raw_data.shape 
    print(raw_data.shape) 
    min_val = np.min(raw_data)
    max_val = np.max(raw_data)

    # Normaliza los datos utilizando la fórmula de normalización min-max
    raw_data = (raw_data - min_val) / (max_val - min_val)'''
    
    if not config.dp.do:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=config.train.warmup_steps, \
                                num_training_steps=(raw_data.shape[0]//config.train.batch_size+1)*config.train.n_epochs)
    

    state = dict(model=model, ema=ema, optimizer=optimizer, step=0)

    if config.setup.global_rank == 0:
        model_parameters = filter(
            lambda p: p.requires_grad, model.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        logging.info('Number of trainable parameters in model: %d' % n_params)
        logging.info('Number of total epochs: %d' % config.train.n_epochs)
        logging.info('Starting training at step %d' % state['step'])
    dist.barrier()
        
    if config.data.name.startswith('mimic'):
        labels = None
        EHR_task = 'binary'
    elif config.data.name.startswith('cinc') or config.data.name.startswith('ecg'):
        EHR_task = 'continuous'
        if config.model.network.use_cfg:
            raw_data = raw_data[:, 1:]
            labels = raw_data[:, 0]
        else:
            labels = None

    class EHRDataset(torch.utils.data.Dataset):
        def __init__(self, data=raw_data, labels=None):
            super().__init__()
            self.data = data
            self.labels = labels

        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, index: int):
            if self.labels is not None:
                return self.data[index], self.labels[index]
            else:
                return self.data[index]

    dataset = EHRDataset(raw_data, labels)
    dataset_loader = torch.utils.data.DataLoader(
        dataset=dataset, shuffle=True, batch_size=config.train.batch_size)

    if config.dp.do:
        privacy_engine = PrivacyEngine()

        # model, optimizer, dataset_loader = privacy_engine.make_private(
        model, optimizer, dataset_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=dataset_loader,
            # noise_multiplier=.7,
            target_delta=config.dp.delta,
            target_epsilon=config.dp.epsilon,
            epochs=config.train.n_epochs,
            max_grad_norm=config.dp.max_grad_norm,
            noise_multiplicity=config.loss.n_noise_samples,
        )
        

    if config.loss.n_classes == 'None':
        config.loss.n_classes = None
    if config.loss.version == 'edm':
        loss_fn = EDMLoss(**config.loss).get_loss
    elif config.loss.version == 'vpsde':
        loss_fn = VPSDELoss(**config.loss).get_loss
    elif config.loss.version == 'vesde':
        loss_fn = VESDELoss(**config.loss).get_loss
    # elif config.loss.version == 'v':
    #     loss_fn = VLoss(**config.loss).get_loss
    else:
        raise NotImplementedError


    if config.sampler.guid_scale == 'None':
        config.sampler.guid_scale = None
    def sampler(x, y=None):
        # if config.sampler.type == 'ddim':
        #     return ddim_sampler(x, y, model, **config.sampler)
        # elif config.sampler.type == 'edm':
        #     return edm_sampler(x, y, model, **config.sampler)
        # else:
        #     raise NotImplementedError
        return ablation_sampler(x, y, model, **config.sampler)


    # snapshot_sampling_shape = (config.sampler.snapshot_batch_size, config.data.resolution)
    snapshot_sampling_shape = (raw_data.shape[0], config.data.resolution)
    if config.data.n_classes == 'None':
        config.data.n_classes = None

    for epoch in range(config.train.n_epochs):
        if config.dp.do:
            with BatchMemoryManager(
                    data_loader=dataset_loader,
                    max_physical_batch_size=config.dp.max_physical_batch_size,
                    optimizer=optimizer,
                    n_splits=config.dp.n_splits if config.dp.n_splits > 0 else None) as memory_safe_data_loader:

                for _, (train) in enumerate(memory_safe_data_loader):

                    if isinstance(train, list):
                        train_x = train[0]
                        train_y = train[1]
                    else:
                        train_x = train
                        train_y = None

                    x = train_x.to(config.setup.device).to(torch.float32)

                    if config.data.n_classes is None:
                        y = None
                    else:
                        y = train_y.to(config.setup.device)
                        if y.dtype == torch.float32:
                            y = y.long()

                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.mean(loss_fn(model, x, y))
                    loss.backward()
                    optimizer.step()

                    if state['step'] % config.train.check_freq == 0 and state['step'] >= config.train.check_freq and config.setup.global_rank == 0:
                        model.eval()
                        with torch.no_grad():
                            ema.store(model.parameters())
                            ema.copy_to(model.parameters())
                            if config.setup.local_rank == 0:
                                sample_random_batch(EHR_task, snapshot_sampling_shape, sampler, 
                                                        sample_dir, config.setup.device, config.data.n_classes)
                            torch.cuda.empty_cache()
                            ema.restore(model.parameters())
                        model.train()

                        logging.info('[%d, %5d] Loss: %.10f' % (epoch+1, state['step'] + 1, loss))
                        syn_data = np.load(sample_dir + '/sample.npy')
                        corr, nzc = plot_dim_dist(raw_data, syn_data, workdir)
                        logging.info('corr: %.4f, none-zero columns: %d'%(corr, nzc)) 
                        logging.info('Eps-value: %.4f' % (privacy_engine.get_epsilon(config.dp.delta)))
                    dist.barrier()


                    if state['step'] % config.train.save_freq == 0 and state['step'] >= config.train.save_freq and config.setup.global_rank == 0:
                        checkpoint_file = os.path.join(
                            checkpoint_dir, 'checkpoint_%d.pth' % state['step'])
                        save_checkpoint(checkpoint_file, state)
                        logging.info(
                            'Saving checkpoint at iteration %d' % state['step'])
                        logging.info('--------------------------------------------')
                    dist.barrier()

                    state['step'] += 1
                    if not optimizer._is_last_step_skipped:
                        state['ema'].update(model.parameters())


        else: # with No Differential Private training
            for _, (train) in enumerate(dataset_loader):

                if isinstance(train, list):
                    train_x = train[0]
                    train_y = train[1]
                else:
                    train_x = train
                    train_y = None

                x = train_x.to(config.setup.device).to(torch.float32)

                if config.data.n_classes is None:
                    y = None
                else:
                    y = train_y.to(config.setup.device)
                    if y.dtype == torch.float32:
                        y = y.long()

                optimizer.zero_grad(set_to_none=True)
                loss = torch.mean(loss_fn(model, x, y))
                # if config.setup.local_rank == 0:
                #     print(loss)
                loss.backward()
                optimizer.step()
                scheduler.step()

                if state['step'] % config.train.check_freq == 0 and state['step'] >= config.train.check_freq and config.setup.local_rank == 0:
                    model.eval()
                    with torch.no_grad():
                        ema.store(model.parameters())
                        ema.copy_to(model.parameters())
                        if config.setup.local_rank == 0:
                            sample_random_batch(EHR_task, snapshot_sampling_shape, sampler, 
                                                    sample_dir, config.setup.device, config.data.n_classes)
                        torch.cuda.empty_cache()
                        ema.restore(model.parameters())
                    model.train()

                    logging.info('[%d, %5d] Loss: %.10f' % (epoch+1, state['step'] + 1, loss))
                    syn_data = np.load(sample_dir + '/sample.npy')
                    corr, nzc = plot_dim_dist(raw_data, syn_data, workdir)
                    logging.info('corr: %.4f, none-zero columns: %d'%(corr, nzc)) 
                dist.barrier()

                if state['step'] % config.train.save_freq == 0 and state['step'] >= config.train.save_freq and config.setup.local_rank == 0:
                    checkpoint_file = os.path.join(
                        checkpoint_dir, 'checkpoint_%d.pth' % state['step'])
                    save_checkpoint(checkpoint_file, state)
                    logging.info(
                        'Saving checkpoint at iteration %d' % state['step'])
                    logging.info('--------------------------------------------')
                dist.barrier()

                state['step'] += 1
                state['ema'].update(model.parameters())

        
    if config.setup.local_rank == 0:
        checkpoint_file = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
        save_checkpoint(checkpoint_file, state)
        logging.info('Saving final checkpoint.')
    dist.barrier()

    model.eval()
    with torch.no_grad():
        ema.store(model.parameters())
        ema.copy_to(model.parameters())

        if config.setup.local_rank == 0:
            logging.info('################################################')
            logging.info('Final Evaluation')
            syn_data = np.load(sample_dir + '/sample.npy')
            corr, nzc = plot_dim_dist(raw_data, syn_data, workdir)
            logging.info('corr: %.4f, none-zero columns: %d'%(corr, nzc)) 
        dist.barrier()

        ema.restore(model.parameters())
        
        
        
config_dict = {
    "setup": {
        "runner": "train_dpdm_base",
        "CUDA_DEVICES": 0,
        "n_gpus_per_node": 1,
        "n_nodes": 1,
        "node_rank": 0,
        "master_address": "127.0.0.1",
        "master_port": 60202,
        "omp_n_threads": 64,
        "workdir_":"/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/train_sp/train_diff",
        "workdir":"/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/train_sp/train_diff",
        "root_folder": "/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/",
         "mode": "train" 
    }, 
    "data": {
        "path_": "/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/train_sp/non_prepo/DATASET_NAME_non_prepo_non_preprocess.pkl",
        "path":"/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/MIMIC_/X_num_train.npy",
        "name": "mimic",
        "resolution": 1782,
        "dataloader_params": {
            "num_workers": 1,
        },
        "n_classes": None,
    },
    "model": {
        "denoiser_name": "edm",
        "denoiser_network": "song",
        "ema_rate": 0.999,
        "params": {
            "sigma_data": 0.14,
            "sigma_min": 0.02,
            "sigma_max": 80.0,
        },
        "network": {
            "z_dim": 1782,
            "time_dim": 384,
            "unit_dims": [1024, 384, 384, 384, 1024],
            "use_cfg": False,
        },
    },
    "optim": {
        "optimizer": "AdamW",
        "params": {
            "lr": 3e-4,
            "weight_decay": 0.0,
        },
    },
    "sampler": {
        "solver": "heun",
        "discretization": "edm",
        "stochastic": False,
        "num_steps": 32,
        "sigma_min": 0.02,
        "sigma_max": 80.0,
        "rho": 7.0,
        "guid_scale": None,
    },
    "train": {
        "seed": 2024,
        "batch_size": 256,
        "warmup_steps": 0,
        "n_epochs": 1,
        "check_freq": 5000,
        "save_freq": 20000,
    },
    "loss": {
        "version": "edm",
        "p_mean": -1.2,
        "p_std": 1.2,
        "sigma_data": 0.14,
        "n_noise_samples": 4,
        "n_classes": None,
    },
    "dp": {
        "do": True,
        "max_grad_norm": 1.1,
        "delta": 1e-5,
        "epsilon": 10.0,
        "max_physical_batch_size": 1024,
        "n_splits": 8,
    },
}
from omegaconf import OmegaConf
config = OmegaConf.create(config_dict)

import logging
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import sys
import argparse

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass


def run_main(config):
    ###
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.setup.CUDA_DEVICES)
    ###
    processes = []
    for rank in range(config.setup.n_gpus_per_node):
        config.setup.local_rank = rank
        config.setup.global_rank = rank + \
            config.setup.node_rank * config.setup.n_gpus_per_node
        config.setup.global_size = config.setup.n_nodes * config.setup.n_gpus_per_node
        # print('Node rank %d, local proc %d, global proc %d' % (
        #     config.setup.node_rank, config.setup.local_rank, config.setup.global_rank))
        p = mp.Process(target=setup, args=(config, main))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def setup(config, fn):
    os.environ['MASTER_ADDR'] = config.setup.master_address
    os.environ['MASTER_PORT'] = '%d' % config.setup.master_port
    os.environ['OMP_NUM_THREADS'] = '%d' % config.setup.omp_n_threads
    torch.cuda.set_device(config.setup.local_rank)
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            rank=config.setup.global_rank,
                            world_size=config.setup.global_size)
    fn(config)
    dist.barrier()
    dist.destroy_process_group()


def set_logger(gfile_stream):
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter(
        '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')


def main(config):
    workdir = os.path.join(config.setup.root_folder, config.setup.workdir)

    if config.setup.mode == 'train':
        if config.setup.global_rank == 0:
            if config.setup.mode == 'train':
                make_dir(workdir)
                gfile_stream = open(os.path.join(workdir, 'stdout.txt'), 'w')
            else:
                if not os.path.exists(workdir):
                    raise ValueError('Working directoy does not exist.')
                gfile_stream = open(os.path.join(workdir, 'stdout.txt'), 'a')

            set_logger(gfile_stream)
            logging.info(config)

        if config.setup.runner == 'train_dpdm_base':
           
            train_dpdm_base(config, workdir, config.setup.mode)
        else:
            raise NotImplementedError('Runner is not yet implemented.')

    elif config.setup.mode == 'eval':
        if config.setup.global_rank == 0:
            make_dir(workdir)
            gfile_stream = open(os.path.join(workdir, 'stdout.txt'), 'w')
            set_logger(gfile_stream)
            logging.info(config)

        if config.setup.runner == 'generate_base':
            from runners import generate_base
            generate_base.evaluation(config, workdir)
        else:
            raise NotImplementedError('Runner is not yet implemented.')


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    
    run_main(config)        