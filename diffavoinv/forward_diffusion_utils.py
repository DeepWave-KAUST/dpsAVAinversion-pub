"""
Forward diffusion utilities
===========================

This module provides helper routines for the forward process in DDPM/DDIM.

Functions
---------
_alpha_vals(scheduler, t)
    Return scaling coefficients sqrt(alpha_t), sqrt(1 - alpha_t).

_forward_diffuse(x0, scheduler, t, noise)
    Apply forward diffusion step:
    q(x_t | x0) = sqrt(alpha_t) * x0 + sqrt(1 - alpha_t) * noise

Notes
-----
These functions are used to implement the forward noising process and to keep
background regions consistent during masked DDIM sampling.
"""

import math
import torch


def _alpha_vals(scheduler, t):
    """
    Return sqrt(alpha_t) and sqrt(1 - alpha_t).

    Parameters
    ----------
    scheduler : diffusers.SchedulerMixin
        Scheduler with attribute ``alphas_cumprod``.
    t : int
        Timestep index.

    Returns
    -------
    sqrt_at : float
        Square root of alpha_t.
    sqrt_1m_at : float
        Square root of 1 - alpha_t.
    """
    a = float(scheduler.alphas_cumprod[int(t)])
    return math.sqrt(a), math.sqrt(max(0.0, 1.0 - a))


@torch.no_grad()
def _forward_diffuse(x0, scheduler, t, noise):
    """
    Apply forward diffusion q(x_t | x0).

    Parameters
    ----------
    x0 : torch.Tensor
        Clean input of shape [B, C, H, W].
    scheduler : diffusers.SchedulerMixin
        Scheduler with attribute ``alphas_cumprod``.
    t : int
        Timestep index.
    noise : torch.Tensor
        Gaussian noise, same shape as x0.

    Returns
    -------
    xt : torch.Tensor
        Noised sample at timestep t, same shape as x0.
    """
    sqrt_at, sqrt_1m_at = _alpha_vals(scheduler, t)
    return sqrt_at * x0 + sqrt_1m_at * noise