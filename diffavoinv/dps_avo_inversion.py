"""
dps_ddim_avo.py

A DPS-guided DDIM sampler for AVO inversion using PyLops:
    G = pylops.avo.prestack.PrestackLinearModelling(..., explicit=False)

Model convention (matches forward modeling snippet exactly):
    m is shaped (nz, 3, nx)
    mvec = m.ravel(order="C")
    dvec = G * mvec
    dPP  = dvec.reshape(nz, ntheta, nx, order="C")

Diffusion convention:
    UNet outputs / tensors shaped (B, C, H, W) where:
        C=3 channels are [log(vp), log(vs), log(rho)] but in NORMALIZED space (typically [-1,1]).
        H = nz, W = nx

Important:
- DPS physics (G) must be applied in PHYSICAL log-space, not normalized space.
- So inside guidance we denormalize x0_pred -> m_log (physical),
  apply G and G.H in numpy, then chain-rule back to normalized space.

Optional EDITING (inv_seq):
- If provide inv_seq and edit_mask: after each DDIM(+DPS) step, overwrite outside-mask with the inverted background trajectory.
- inv_seq is the DDIM inversion trajectory of initial (m0) model, with shape [T,B,3,nz,nx]
  where T matches num_inference_steps (same scheduler.timesteps).
"""

import math
from typing import Optional, Literal

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ------------------------------------------------------------
# Helpers (small and explicit)
# ------------------------------------------------------------
def _to_bchw(x: torch.Tensor) -> torch.Tensor:
    """Force input to shape (B,C,H,W)."""
    if x.dim() == 2:   # (H,W)
        return x.unsqueeze(0).unsqueeze(0)
    if x.dim() == 3:   # (B,H,W)
        return x.unsqueeze(1)
    return x           # assume already (B,C,H,W)


def _denorm_m11_to_phys(x_norm: torch.Tensor,
                        vmin: torch.Tensor,
                        vmax: torch.Tensor) -> torch.Tensor:
    """
    Inverse of per-channel min-max normalization to [-1,1].

    If training did:
        x_norm = 2*(x - vmin)/(vmax - vmin) - 1
    then:
        x = (x_norm+1)/2*(vmax-vmin) + vmin

    x_norm: (B,3,nz,nx)
    vmin/vmax: (3,) torch
    """
    vmin = vmin.view(1, -1, 1, 1)
    vmax = vmax.view(1, -1, 1, 1)
    return (x_norm + 1.0) * 0.5 * (vmax - vmin) + vmin


def _dm_dx0_minmax(vmin: torch.Tensor, vmax: torch.Tensor) -> torch.Tensor:
    """
    Chain rule derivative for normalized DPS gradient: ∂L/∂m = ∂L/∂x0 ∂x0/∂m
    ∂L/∂m = g_norm, normalized gradient 
    ∂L/∂x0 = g_torch physical gradient 
    ∂x0/∂m = scaling between physics and norm
    
    Computes ∂x0/∂m for per-sample min–max normalization to [-1, 1].

    Args:
        vmin: (C,) tensor of per-channel minima
        vmax: (C,) tensor of per-channel maxima

    Returns:
        (1, C, 1, 1) tensor suitable for broadcasting over (B, C, H, W)
    """

    return (vmax - vmin).view(1, -1, 1, 1) * 0.5


def avo_phys_grad_from_mlog(
    m_log_b: torch.Tensor,     # (3,nz,nx) torch, PHYSICAL log-space
    G,                         # PyLops operator (explicit=False)
    d_obs_vec: np.ndarray,     # (nz*ntheta*nx,) flattened C-order
) -> torch.Tensor:
    """
    Compute physical gradient in log-space:
        g = G.H (d_obs - G m)

    IMPORTANT: This matches the convention:
        m is (nz,3,nx)
        mvec = m.ravel('C')
        dvec = G*mvec
        dPP  = dvec.reshape(nz,ntheta,nx,'C')
    """
    C, nz, nx = m_log_b.shape
    assert C == 3, "Expected 3 channels: logvp, logvs, logrho"

    # (3,nz,nx) -> (nz,3,nx) because your m is (nz,3,nx)
    m_pylops = ( m_log_b.permute(1, 0, 2).detach().cpu().numpy().astype(np.float32, copy=False) )  # (nz,3,nx)

    # exact same flattening as in forward modelling
    mvec = m_pylops.ravel(order="C")  # (3*nz*nx,)

    # forward model in vector form
    d_pred_vec = (G * mvec).astype(np.float32, copy=False).ravel(order="C")

    # residual (data misfit)
    r = (d_obs_vec - d_pred_vec).astype(np.float32, copy=False)

    # adjoint G.H maps residual to gradient in model space (vector form)
    gvec = (G.H * r).astype(np.float32, copy=False)  # (3*nz*nx,)

    # reshape back to model layout (nz,3,nx)
    g_pylops = gvec.reshape(nz, 3, nx, order="C")

    # back to torch diffusion layout (3,nz,nx)
    g_torch = ( torch.from_numpy(g_pylops).permute(1, 0, 2).to(device=m_log_b.device, dtype=m_log_b.dtype) )

    return g_torch  # (3,nz,nx)


# ------------------------------------------------------------
# Main DPS-DDIM for AVO (with optional inv_seq editing)
# ------------------------------------------------------------
def dps_ddim_avo(
    unet: torch.nn.Module,
    scheduler,
    *,
    # --- model input (normalized) ---
    m0_norm: torch.Tensor,                  # (B,3,nz,nx) normalized in [-1,1]

    # --- AVO physics ---
    G,                                      # pylops.avo.prestack.PrestackLinearModelling(..., explicit=False)
    d_angle_gathers: np.ndarray,            # observed data (nz, ntheta, nx) or equivalent

    # --- normalization stats for log(vp),log(vs),log(rho) ---
    vmin: np.ndarray,                       # (3,) physical mins (in log-space)
    vmax: np.ndarray,                       # (3,) physical maxs (in log-space)

    # --- sampling ---
    num_inference_steps: int = 50,
    eta: float = 0.0,                       # 0 = deterministic DDIM, >0 stochastic

    # --- guidance ---
    alpha_guidance: float = 0.2,
    last_frac: float = 0.3,
    rms_clip: Optional[float] = 0.25,

    # --- warm start controls ---
    warm_start_bg: Optional[torch.Tensor] = None,   # (B,3,nz,nx) normalized [-1,1]
    t_start_frac: float = 1.0,                      # 1.0 = start from early step; smaller = start later
    noise_mask: Optional[torch.Tensor] = None,      # (nz,nx) or (B,1,nz,nx) or (B,3,nz,nx); 1=noise, 0=no-noise

    # --- optional mask: 1 = allow DPS updates, 0 = protect ---
    edit_mask: Optional[torch.Tensor] = None,   # (H,W) or (B,1,H,W)

    # --- OPTIONAL EDITING via inversion sequence ---
    inv_seq: Optional[torch.Tensor] = None,     # [T,B,3,nz,nx] inverted background trajectory
    hard_edit: bool = True,                     # if True, overwrite outside-mask with inv_seq

    # --- jacobian mode ---
    jacobian_mode: Literal["full", "identity"] = "full",

    # --- device ---
    device: str = "cuda",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Returns:
        x_final_norm: (B,3,nz,nx) final sample in NORMALIZED space [-1,1].

    Editing behavior:
      - If inv_seq is provided AND edit_mask is provided AND hard_edit=True:
            after each DDIM(+DPS) update at timestep t, we enforce:
                x <- edit_mask*x + (1-edit_mask)*inv_seq[t]
        This keeps background fixed outside the mask across the entire trajectory.
    """
    assert jacobian_mode in ("full", "identity")

    # -------------------------
    # 0) Prepare shapes + constants
    # -------------------------
    x0 = _to_bchw(m0_norm).to(device=device, dtype=torch.float32)  # (B,3,nz,nx)
    B, C, nz, nx = x0.shape
    assert C == 3, f"Expected 3 channels, got C={C}"

    # observed data: force same convention (nz,ntheta,nx) then ravel('C')
    d_obs = np.asarray(d_angle_gathers, dtype=np.float32)
    if d_obs.ndim == 3:
        assert d_obs.shape[0] == nz and d_obs.shape[2] == nx, \
            f"d_obs shape {d_obs.shape} not compatible with (nz,*,nx)=({nz},*,{nx})"
        ntheta = d_obs.shape[1]
        d_obs_vec = d_obs.reshape(nz, ntheta, nx, order="C").ravel(order="C")
    else:
        d_obs_vec = d_obs.ravel(order="C")
        ntheta = None

    # normalization tensors (log-space mins/maxs) for g_torch / g_norm scaling:
    vmin_t = torch.as_tensor(vmin, dtype=torch.float32, device=device)
    vmax_t = torch.as_tensor(vmax, dtype=torch.float32, device=device)
    dm_dx0 = _dm_dx0_minmax(vmin_t, vmax_t)  # (1,3,1,1)

    # -------------------------
    # mask handling (broadcast to (B,3,nz,nx))
    # -------------------------
    if edit_mask is not None:
        m = _to_bchw(edit_mask).to(device=device, dtype=torch.float32)  # (B,1,nz,nx) or (1,1,nz,nx)
        if m.shape[-2:] != (nz, nx):
            m = F.interpolate(m, size=(nz, nx), mode="nearest")
        if m.shape[0] == 1 and B > 1:
            m = m.expand(B, -1, -1, -1)
        if m.shape[1] == 1:
            m = m.expand(B, C, nz, nx)
        edit_mask = m
    else:
        # no mask -> cannot do "hard editing"
        if inv_seq is not None:
            raise ValueError("inv_seq provided but edit_mask is None. Provide edit_mask to enable editing.")

    # -------------------------
    # 1) Setup DDIM time schedule
    # -------------------------
    unet.eval()
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps  # descending
    T = len(timesteps)
    
    # -------------------------
    # 1a) choose warm-start index k_start (must be before k_on)
    # -------------------------
    t_start_frac = float( max(0.0, min(1.0, t_start_frac)) )
    k_start = int((1.0 - t_start_frac) * T)
    k_start = max(0, min(k_start, T - 1))


    # alphas_cumprod: diffusion schedule constants
    alphas_cum = scheduler.alphas_cumprod.to(device)

    # guidance only in the last part of the *remaining* trajectory
    k_on = k_start + max(0, min(int((1.0 - float(last_frac)) * (T - k_start)), (T - 1 - k_start)) )


    # -------------------------
    # 1b) Optional: prepare inv_seq for editing
    # -------------------------
    t_to_inv_idx = {}
    if inv_seq is not None:
        # inv_seq must be [T,B,3,nz,nx] where T matches this scheduler call
        if inv_seq.dim() != 5:
            raise ValueError("inv_seq must have shape [T,B,3,nz,nx].")
        if inv_seq.shape[0] != T:
            raise ValueError(f"inv_seq length {inv_seq.shape[0]} must equal T={T} (num_inference_steps).")
        if inv_seq.shape[2] != 3:
            raise ValueError(f"inv_seq channels must be 3, got {inv_seq.shape[2]}.")

        # expand batch if needed
        if inv_seq.shape[1] == 1 and B > 1:
            inv_seq = inv_seq.expand(-1, B, -1, -1, -1)

        # resize spatial if needed
        if inv_seq.shape[-2:] != (nz, nx):
            inv_seq = F.interpolate(
                inv_seq.reshape(T * inv_seq.shape[1], inv_seq.shape[2], inv_seq.shape[3], inv_seq.shape[4]),
                size=(nz, nx),
                mode="bilinear",
                align_corners=False
            ).reshape(T, inv_seq.shape[1], inv_seq.shape[2], nz, nx)

        # build map timestep_value -> inv_seq index (same logic you used before)
        with torch.no_grad():
            t_vals_np = timesteps.detach().cpu().numpy().astype(int)  # [T] descending
            rev_ts_np = t_vals_np[::-1]                               # ascending
            t_to_inv_idx = {int(rev_ts_np[k]): k for k in range(T)}

    # -------------------------
    # 2) Initialize x at timestep t_start (warm start)
    # -------------------------

    t_start = timesteps[k_start]
    t_start_int = int(t_start.item())

    a0 = alphas_cum[t_start_int].sqrt().view(1, 1, 1, 1)
    b0 = (1.0 - alphas_cum[t_start_int]).sqrt().view(1, 1, 1, 1)

    # choose background (normalized)
    if warm_start_bg is not None:
        bg = _to_bchw(warm_start_bg).to(device=device, dtype=torch.float32)
        if bg.shape[-2:] != (nz, nx):
            bg = F.interpolate(bg, size=(nz, nx), mode="bilinear", align_corners=False)
        if bg.shape[0] == 1 and B > 1:
            bg = bg.expand(B, -1, -1, -1)
    else:
        bg = x0  # default background is m0_norm

    # starting noise (generator-safe)
    if generator is not None:
        eps0 = torch.randn(bg.shape, device=bg.device, dtype=bg.dtype, generator=generator)
    else:
        eps0 = torch.randn_like(bg)

    # optional: restrict where noise is injected
    if noise_mask is not None:
        nm = _to_bchw(noise_mask).to(device=device, dtype=torch.float32)
        if nm.shape[-2:] != (nz, nx):
            nm = F.interpolate(nm, size=(nz, nx), mode="nearest")
        if nm.shape[0] == 1 and B > 1:
            nm = nm.expand(B, -1, -1, -1)
        # make nm have 3 channels
        if nm.shape[1] == 1:
            nm = nm.expand(B, C, nz, nx)
        elif nm.shape[1] != C:
            raise ValueError(f"noise_mask channels must be 1 or {C}, got {nm.shape[1]}")
        eps0 = eps0 * nm

    x = (a0 * bg + b0 * eps0).clamp(-1.0, 1.0)


    # -------------------------
    # 3) Main DDIM loop
    # -------------------------
    for k in tqdm(range(k_start, T - 1), desc="DPS-DDIM AVO"):

        # detach because we rebuild the graph each step (only if jacobian_mode='full')
        x = x.detach()
        if jacobian_mode == "full":
            x.requires_grad_(True)

        t = timesteps[k]
        t_prev = timesteps[k + 1]
        t_int = int(t.item())
        t_prev_int = int(t_prev.item())

        # timestep as batch vector
        t_b = torch.full((B,), t_int, dtype=torch.long, device=device)

        # -------------------------
        # (A) UNet predicts noise eps_theta(x_t, t)
        # -------------------------
        x_in = scheduler.scale_model_input(x, t_b)
        noise_pred = unet(x_in, t_b).sample  # (B,3,nz,nx)

        # -------------------------
        # (B) Predict x0 at this step (normalized "clean" estimate)
        # -------------------------
        alpha_t = float(scheduler.alphas_cumprod[t_int])
        alpha_prev = float(scheduler.alphas_cumprod[t_prev_int])

        sqrt_alpha_t = math.sqrt(alpha_t)
        sqrt_1m_alpha_t = math.sqrt(max(0.0, 1.0 - alpha_t))

        x0_pred = (x - sqrt_1m_alpha_t * noise_pred) / (sqrt_alpha_t + 1e-12)
        x0_pred = x0_pred.clamp(-1.0, 1.0)

        # -------------------------
        # (C) DPS guidance (AVO physics) only near the end
        # -------------------------
        guidance = None
        if (k >= k_on) and (alpha_guidance > 0.0):

            # 1) denormalize -> PHYSICAL log-space (this is what G expects)
            m_log = _denorm_m11_to_phys(x0_pred, vmin_t, vmax_t)  # (B,3,nz,nx)

            # 2) compute physical gradient g = G.H (d_obs - G*m) for each sample in batch
            g_phys_list = []
            for b in range(B):
                g_phys_b = avo_phys_grad_from_mlog(
                    m_log[b],     # (3,nz,nx) physical log-space
                    G,
                    d_obs_vec,
                )
                g_phys_list.append(g_phys_b.unsqueeze(0))

            g_phys = torch.cat(g_phys_list, dim=0)  # (B,3,nz,nx)

            # 3) chain rule to get gradient in NORMALIZED x0 space
            #    dm/dx0 = (vmax - vmin)/2 for min-max normalization to [-1,1]
            dX = g_phys * dm_dx0  # (B,3,nz,nx)

            # 4) optional mask (protect region from DPS)
            if edit_mask is not None:
                dX = dX * edit_mask

            # 5) RMS clip (stability)
            if (rms_clip is not None) and (rms_clip > 0):
                rms = dX.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-8)
                dX = dX * torch.minimum(
                    torch.ones_like(rms),
                    torch.as_tensor(rms_clip, device=device, dtype=x.dtype) / rms
                )

            # 6) convert dX (x0-space) to a guidance direction in x_t space
            if jacobian_mode == "full":
                # full DPS: backprop through x0_pred(x_t)
                unet.zero_grad(set_to_none=True)
                if x.grad is not None:
                    x.grad.zero_()

                # choose scalar loss so that dL/dx0_pred = -dX
                L = -(x0_pred * dX).sum()
                L.backward()
                guidance = x.grad.detach()  # (B,3,nz,nx)

            else:
                # identity-J approximation
                guidance = dX

        # -------------------------
        # (D) DDIM prior step (x_t -> x_{t_prev})
        # -------------------------
        if eta > 0.0:
            sigma_t = eta * math.sqrt(
                (1.0 - alpha_prev) / (1.0 - alpha_t) *
                (1.0 - alpha_t / alpha_prev)
            )
        else:
            sigma_t = 0.0

        coeff_eps = math.sqrt(max(0.0, 1.0 - alpha_prev - sigma_t**2))

        # stochastic part for eta>0 (generator-safe)
        if sigma_t > 0.0:
            if generator is not None:
                z = torch.randn(
                    x.shape,
                    device=x.device,
                    dtype=x.dtype,
                    generator=generator,
                )
            else:
                z = torch.randn_like(x)
        else:
            z = 0.0

        # what DDIM would do without physics
        x_prior = (
            math.sqrt(alpha_prev) * x0_pred.detach()
            + coeff_eps * noise_pred.detach()
            + sigma_t * z
        )

        # -------------------------
        # (E) Apply DPS correction in x_t space
        # -------------------------
        if guidance is not None:
            x = (x_prior - alpha_guidance * guidance).clamp(-1.0, 1.0)
        else:
            x = x_prior.clamp(-1.0, 1.0)

        # -------------------------
        # (F) OPTIONAL HARD EDITING using inv_seq (keep background outside mask)
        # -------------------------
        if hard_edit and (inv_seq is not None) and (edit_mask is not None):
            # enforce outside-mask at the *current* timestep t_int
            if t_int in t_to_inv_idx:
                x_bg_t = inv_seq[t_to_inv_idx[t_int]].to(device=x.device, dtype=x.dtype)  # (B,3,nz,nx)
                x = edit_mask * x + (1.0 - edit_mask) * x_bg_t

    # final normalized sample
    return x.detach()
