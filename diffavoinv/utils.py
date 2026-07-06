import numpy as np
from typing import Tuple


def normalize_m11_logmodel(
    m: np.ndarray,
    vmin: np.ndarray,
    vmax: np.ndarray,
) -> np.ndarray:
    """
    Min–max normalize a log-parameter model to [-1, 1], per channel.

    Parameters
    ----------
    m : np.ndarray
        Physical log-model with shape (nz, 3, nx),
        channels = [log(vp), log(vs), log(rho)].
    vmin : np.ndarray
        Per-channel minimum values, shape (3,).
    vmax : np.ndarray
        Per-channel maximum values, shape (3,).

    Returns
    -------
    m_norm : np.ndarray
        Normalized model in [-1,1], shape (nz, 3, nx).
    """
    assert m.ndim == 3 and m.shape[1] == 3, f"Expected (nz,3,nx), got {m.shape}"
    assert vmin.shape == (3,) and vmax.shape == (3,)

    m_norm = np.empty_like(m, dtype=np.float32)

    for ip in range(3):
        m_norm[:, ip, :] = (
            2.0 * (m[:, ip, :] - vmin[ip]) / (vmax[ip] - vmin[ip]) - 1.0
        )

    return m_norm



def denormalize_m11_logmodel(
    m_norm: np.ndarray,
    vmin: np.ndarray,
    vmax: np.ndarray,
) -> np.ndarray:
    """
    Inverse of min–max normalization from [-1,1] back to physical log-space.

    Parameters
    ----------
    m_norm : np.ndarray
        Normalized model in [-1,1], shape (nz, 3, nx).
    vmin : np.ndarray
        Per-channel minimum values, shape (3,).
    vmax : np.ndarray
        Per-channel maximum values, shape (3,).

    Returns
    -------
    m : np.ndarray
        Physical log-model, shape (nz, 3, nx).
    """
    assert m_norm.ndim == 3 and m_norm.shape[1] == 3, f"Expected (nz,3,nx), got {m_norm.shape}"
    assert vmin.shape == (3,) and vmax.shape == (3,)

    m = np.empty_like(m_norm, dtype=np.float32)

    for ip in range(3):
        m[:, ip, :] = (
            0.5 * (m_norm[:, ip, :] + 1.0) * (vmax[ip] - vmin[ip]) + vmin[ip]
        )

    return m
