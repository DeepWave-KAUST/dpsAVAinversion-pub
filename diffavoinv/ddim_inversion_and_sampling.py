import torch
import math
from tqdm import tqdm
from typing import Optional, Callable
from diffavoinv.forward_diffusion_utils import _alpha_vals, _forward_diffuse

# ------------------------------------------------------------------------------------------------------------
# Unconditional Sampling 
# ------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def sample_unconditional(
    unet: torch.nn.Module,
    scheduler,
    num_inference_steps: int = 50,
    device: str = "cuda",
    shape: tuple = (1, 1, 64, 64),
    generator: Optional[torch.Generator] = None # generator: torch.Generator | None = None if python 3.10
) -> torch.Tensor:
    """
    Unconditional DDIM sampling (eta=0) from pure Gaussian noise.

    This routine generates a new sample x_0 without conditioning, starting from
    white noise and following the scheduler's timesteps. It mirrors the style
    used in the other utilities (UNet noise prediction via `scale_model_input`,
    `scheduler.step` update per step).

    Parameters
    ----------
    unet : torch.nn.Module
        Diffusion UNet predicting noise eps = eps_theta(x_t, t). Must accept
        inputs shaped like `shape` and return an object with `.sample`.
    scheduler : diffusers.SchedulerMixin
        Diffusers scheduler compatible with DDIM-like updates; must provide
        `set_timesteps`, `timesteps`, `scale_model_input`, and `step`.
        Configure the scheduler with eta=0 if applicable for deterministic DDIM.
    num_inference_steps : int, optional
        Number of sampling steps. Default is 50.
    device : str, optional
        Torch device for computation. Default is "cuda".
    shape : tuple, optional
        Output tensor shape as (B, C, H, W). Default is (1, 1, 64, 64).
    generator : torch.Generator or None, optional
        Optional torch RNG generator for reproducible sampling.

    Returns
    -------
    x : torch.Tensor
        Generated sample at t=0 with shape `shape`, scaled as per training
        (e.g., typically in [-1, 1]).

    Notes
    -----
    - Uses `scheduler.scale_model_input(x, t)` before UNet evaluation.
    - Uses `scheduler.step(noise_pred, t, x).prev_sample` to step t -> t-1.
    - For fully deterministic DDIM, ensure the scheduler is configured with
      eta=0 (if the scheduler supports it).
    """
    unet.eval()
    scheduler.set_timesteps(num_inference_steps, device=device)

    # Start from pure Gaussian noise
    x = torch.randn(*shape, device=device, generator=generator)

    # Descending timesteps as provided by Diffusers
    for t in scheduler.timesteps:
        # Model prediction (noise)
        t_tensor = torch.full((shape[0],), int(t), dtype=torch.long, device=device)
        x_in = scheduler.scale_model_input(x, t_tensor)
        noise_pred = unet(x_in, t_tensor).sample

        # Scheduler update: x_{t-1}
        out = scheduler.step(noise_pred, t, x)
        x = out.prev_sample

    return x



# ------------------------------------------------------------------------------------------------------------
# DDIM Inversion 
# ------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def invert_ddim_pixel(x0, unet, scheduler, num_inference_steps=50, device="cuda"):
    
    """
    Invert clean inputs x0 to a terminal latent x_T (deterministic DDIM, eta=0).

    Parameters
    ----------
    x0 : torch.Tensor
        Clean inputs of shape [B, C, H, W], scaled as in training (e.g., [-1, 1]).
    unet : torch.nn.Module
        Diffusion UNet predicting noise eps = eps_theta(x_t, t).
    scheduler : diffusers.SchedulerMixin
        Scheduler compatible with DDIM; must provide `alphas_cumprod`,
        `set_timesteps`, and `scale_model_input`.
    num_inference_steps : int, optional
        Number of inference steps across the schedule. Default is 50.
    device : str, optional
        Torch device for computation. Default is "cuda".

    Returns
    -------
    seq : torch.Tensor
        Latent trajectory of shape [num_steps, B, C, H, W], from x_0 to x_T.
        The last element `seq[-1]` is the terminal latent x_T.

    Notes
    -----
    Useful for creating a background-consistent x_T that matches the
    UNet+Scheduler pairing before conditional or masked sampling.
    
    Examples
    --------
    >>> inv = invert_ddim_pixel(x0=background, unet=unet, scheduler=sched, num_inference_steps=50)
    >>> xT_bg = inv[-1]
    """
    
    unet.eval()
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps.cpu().numpy().astype(int)  # descending

    rev_ts = timesteps[::-1]  # small->large
    x = x0.clone().to(device)
    seq = [x.clone()]

    B = x.shape[0]
    for i in tqdm(range(len(rev_ts) - 1), desc="Inverting (x0->xT)"):
        t_prev = int(rev_ts[i])      # smaller
        t = int(rev_ts[i + 1])       # larger

        t_tensor = torch.full((B,), t, dtype=torch.long, device=device)
        x_in = scheduler.scale_model_input(x, t_tensor)
        eps = unet(x_in, t_tensor).sample

        sqrt_at, sqrt_1m_at = _alpha_vals(scheduler, t)
        sqrt_aprev, sqrt_1m_aprev = _alpha_vals(scheduler, t_prev)

        # Inverted DDIM step
        x = (x - sqrt_1m_aprev * eps) * (sqrt_at / (sqrt_aprev + 1e-12)) + sqrt_1m_at * eps
        seq.append(x.clone())

    return torch.stack(seq, dim=0)  # [num_steps, B, C, H, W]


# ------------------------------------------------------------------------------------------------------------
# Sampling from xT 
# ------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def ddim_sample_from_xT(
    xT: torch.Tensor,
    unet: torch.nn.Module,
    scheduler,
    num_inference_steps: int = 50,
    device: str = "cuda",
    eta: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    DDIM sampling from a terminal latent x_T back to x_0.

    If eta = 0.0  -> deterministic DDIM (ODE).
    If eta > 0.0  -> stochastic sampling (SDE-like), exploring around the inverted latent.

    Parameters
    ----------
    xT : [B, C, H, W]
        Starting latent (typically from inversion).
    unet : torch.nn.Module
        Diffusion UNet predicting noise eps = eps_theta(x_t, t).
    scheduler : diffusers.DDIMScheduler (or compatible)
        Must expose `alphas_cumprod`, `set_timesteps`, `timesteps`,
        and `scale_model_input`.
    eta : float
        Stochasticity parameter. 0 = fully deterministic DDIM,
        >0 adds noise at each step.

    Returns
    -------
    x0 : [B, C, H, W]
        Sample at t=0.
    """
    unet = unet.to(device)
    x = xT.to(device).clone()
    B = x.shape[0]

    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps.cpu().numpy().astype(int)

    for i in tqdm(range(len(timesteps) - 1), desc="DDIM sampling (xT->x0)"):
        t = int(timesteps[i])          # current (larger)
        t_prev = int(timesteps[i + 1]) # next (smaller)

        # 1) Predict noise at step t
        t_tensor = torch.full((B,), t, dtype=torch.long, device=device)
        model_in = scheduler.scale_model_input(x, t_tensor)
        noise_pred = unet(model_in, t_tensor).sample

        # 2) Extract alpha_t, alpha_t_prev
        alpha_t = float(scheduler.alphas_cumprod[t])
        alpha_prev = float(scheduler.alphas_cumprod[t_prev])

        sqrt_alpha_t = math.sqrt(alpha_t)
        sqrt_1m_alpha_t = math.sqrt(max(0.0, 1.0 - alpha_t))

        # 3) Predict x0 at step t
        x0_pred = (x - sqrt_1m_alpha_t * noise_pred) / (sqrt_alpha_t + 1e-12)

        # 4) Compute sigma_t for general DDIM (controls stochasticity)
        if eta > 0.0:
            sigma_t = eta * math.sqrt(
                (1.0 - alpha_prev) / (1.0 - alpha_t)
                * (1.0 - alpha_t / alpha_prev)
            )
        else:
            sigma_t = 0.0

        # Coefficient in front of eps_theta term
        coeff_eps = math.sqrt(max(0.0, 1.0 - alpha_prev - sigma_t**2))

        # 5) Sample fresh Gaussian noise if needed
        if sigma_t > 0.0:
            if generator is not None:
                z = torch.randn_like(x, generator=generator)
            else:
                z = torch.randn_like(x)
        else:
            z = 0.0

        # 6) Final update to x_{t_prev}
        x = (
            math.sqrt(alpha_prev) * x0_pred
            + coeff_eps * noise_pred
            + sigma_t * z
        )

    return x