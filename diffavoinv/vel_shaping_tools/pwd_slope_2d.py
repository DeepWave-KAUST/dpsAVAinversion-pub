
import numpy as np
import numpy.typing as npt
from typing import Tuple
import pylops
from pylops.utils.typing import NDArray




def triangular_smoothing_from_boxcars(
    nsmooth: Tuple[int, int],  # user-facing "target triangle base lengths"
    dims: Tuple[int, int],
    dtype: str = "float32",
):
    r"""
    Triangular smoothing operator built from two successive boxcar smoothers.

    This operator is constructed as the composition of two separable 2D boxcar
    smoothers, i.e. :math:`S_\triangle = S_\text{box}^T S_\text{box}`.
    Applying two passes of a length-:math:`L` boxcar is equivalent to applying
    a triangular kernel of base length :math:`(2L-1)`.

    If the requested ``nsmooth`` base length is even, the closest realizable
    triangular length :math:`2L-1 \leq \text{nsmooth}` is chosen.

    Parameters
    ----------
    nsmooth : :obj:`tuple` of :obj:`int`
        Desired base lengths of the triangular kernel along the two axes
        (``ny``, ``nx``).
    dims : :obj:`tuple` of :obj:`int`
        Dimensions of the input model (``nz``, ``nx``).
    dtype : :obj:`str`, optional
        Data type of the operator. Default is ``'float32'``.

    Returns
    -------
    Sop : :obj:`pylops.LinearOperator`
        Triangular smoothing operator of shape ``(nz*nx, nz*nx)``.

    Notes
    -----
    The realized triangular base lengths are approximately
    :math:`(2L_y-1, 2L_x-1)` where
    :math:`L_y = \lfloor(\text{ny}+1)/2\rfloor` and
    :math:`L_x = \lfloor(\text{nx}+1)/2\rfloor`.

    Examples
    --------
    >>> import numpy as np
    >>> import pylops
    >>> from pylops.utils import dottest
    >>> Sop = triangular_smoothing_from_boxcars(nsmooth=(7, 7), dims=(10, 10))
    >>> x = np.random.randn(100)
    >>> y = Sop * x
    >>> dottest(Sop, 100, 100, complexflag=False)
    True
    """
    ny, nx = nsmooth
    # two passes of a length-L boxcar produce a triangle of length (2*L - 1)
    Ly = (ny + 1) // 2  # closest integer L so that 2*Ly-1 <= ny
    Lx = (nx + 1) // 2

    Sbox = pylops.Smoothing2D(nsmooth=(Ly, Lx), dims=dims, dtype=dtype)
    Sop  = Sbox @ Sbox
    return Sop



# -------------------------------
# NUMBA-accelerated core
# -------------------------------
try:
    from numba import njit, prange
except ImportError:
    # graceful fallback if numba isn't available
    def njit(*args, **kwargs):
        def wrap(f): return f
        return wrap
    def prange(*args, **kwargs):
        return range(*args, **kwargs)

@njit(fastmath=True, cache=True)
def _B3(sigma: float):
    # 3 taps
    b0 = (1.0 - sigma) * (2.0 - sigma) / 12.0
    b1 = (2.0 + sigma) * (2.0 - sigma) / 6.0
    b2 = (1.0 + sigma) * (2.0 + sigma) / 12.0
    return b0, b1, b2

@njit(fastmath=True, cache=True)
def _B3d(sigma: float):
    # derivatives (3 taps)
    b0 = -(2.0 - sigma) / 12.0 - (1.0 - sigma) / 12.0
    b1 = (2.0 - sigma) / 6.0 - (2.0 + sigma) / 6.0
    b2 = (2.0 + sigma) / 12.0 + (1.0 + sigma) / 12.0
    return b0, b1, b2

@njit(fastmath=True, cache=True)
def _B5(sigma: float):
    # 5 taps
    s = sigma
    b0 = (1-s)*(2-s)*(3-s)*(4-s)/1680.0
    b1 = (4-s)*(2-s)*(3-s)*(4+s)/420.0
    b2 = (4-s)*(3-s)*(3+s)*(4+s)/280.0
    b3 = (4-s)*(2+s)*(3+s)*(4+s)/420.0
    b4 = (1+s)*(2+s)*(3+s)*(4+s)/1680.0
    return b0, b1, b2, b3, b4

@njit(fastmath=True, cache=True)
def _B5d(sigma: float):
    s = sigma
    b0 = -((2-s)*(3-s)*(4-s) + (1-s)*(3-s)*(4-s) + (1-s)*(2-s)*(4-s) + (1-s)*(2-s)*(3-s))/1680.0
    b1 = -((2-s)*(3-s)*(4+s) + (4-s)*(3-s)*(4+s) + (4-s)*(2-s)*(4+s))/420.0 + (4-s)*(2-s)*(3-s)/420.0
    b2 = -((3-s)*(3+s)*(4+s) + (4-s)*(3+s)*(4+s))/280.0 + (4-s)*(3-s)*(4+s)/280.0 + (4-s)*(3-s)*(3+s)/280.0
    b3 = -((2+s)*(3+s)*(4+s))/420.0 + (4-s)*(3+s)*(4+s)/420.0 + (4-s)*(2+s)*(4+s)/420.0 + (4-s)*(2+s)*(3+s)/420.0
    b4 = ((2+s)*(3+s)*(4+s) + (1+s)*(3+s)*(4+s) + (1+s)*(2+s)*(4+s) + (1+s)*(2+s)*(3+s))/1680.0
    return b0, b1, b2, b3, b4

@njit(parallel=True, fastmath=True, cache=True)
def conv_allpass_numba(din: np.ndarray, dip: np.ndarray, order: int, u1: np.ndarray, u2: np.ndarray):
    r"""
    Plane-wave destruction all-pass convolution (Numba accelerated).

    Compute numerator and denominator residuals for local slope estimation
    using plane-wave destruction filters. The routine fills the provided
    output arrays ``u1`` and ``u2`` in place.

    Parameters
    ----------
    din : :obj:`numpy.ndarray`
        Input 2D array of shape ``(n1, n2)`` containing the data
        (e.g., seismic section).
    dip : :obj:`numpy.ndarray`
        Local slope field of shape ``(n1, n2)``, in samples per trace
        (:math:`dz/dx`).
    order : :obj:`int`
        Filter accuracy order. Use ``1`` for a 3-tap filter
        (B3 polynomial) or ``2`` for a 5-tap filter (B5 polynomial).
    u1 : :obj:`numpy.ndarray`
        Output array of shape ``(n1, n2)``. Filled with the denominator
        residual :math:`C'(\sigma)\,d`.
    u2 : :obj:`numpy.ndarray`
        Output array of shape ``(n1, n2)``. Filled with the numerator
        residual :math:`C(\sigma)\,d`.

    Notes
    -----
    The all-pass filters implement the plane-wave destruction operator
    described in Fomel (2002). For each spatial location, the method
    evaluates the filter stencil according to the local slope
    ``dip[i1, i2]`` and accumulates contributions across the stencil
    window. Coefficients are obtained from the corresponding B-spline
    polynomials:

    - Order 1: 3-tap filter (B3, quadratic).
    - Order 2: 5-tap filter (B5, quartic).

    This routine is the core kernel used inside :func:`pwddip2d`.

    References
    ----------
    - Fomel, S. (2002). "Applications of plane-wave destruction filters."
      *Geophysics*, 67(6), 1946-1960.

    Examples
    --------
    >>> import numpy as np
    >>> n1, n2 = 50, 30
    >>> din = np.random.randn(n1, n2).astype('float32')
    >>> dip = np.zeros_like(din, dtype='float32')
    >>> u1 = np.zeros_like(din)
    >>> u2 = np.zeros_like(din)
    >>> conv_allpass_numba(din, dip, order=1, u1=u1, u2=u2)
    >>> u1.shape, u2.shape
    ((50, 30), (50, 30))
    """
    n1, n2 = din.shape
    nw = 1 if order == 1 else 2

    # zero outputs (safer if caller reuses buffers)
    for j in prange(n1):
        for i in range(n2):
            u1[j, i] = 0.0
            u2[j, i] = 0.0

    # loop over valid stencil region
    for i1 in prange(nw, n1 - nw):
        for i2 in range(0, n2 - 1):
            s = dip[i1, i2]

            if order == 1:
                b0d, b1d, b2d = _B3d(s)
                b0,  b1,  b2  = _B3(s)
                # taps mapping iw=-1,0,1  -> indices 0,1,2
                # accumulate over window
                # iw = -1
                v = din[i1 - 1, i2 + 1] - din[i1 + 1, i2]
                u1[i1, i2] += v * b0d
                u2[i1, i2] += v * b0
                # iw = 0
                v = din[i1 + 0, i2 + 1] - din[i1 + 0, i2]
                u1[i1, i2] += v * b1d
                u2[i1, i2] += v * b1
                # iw = +1
                v = din[i1 + 1, i2 + 1] - din[i1 - 1, i2]
                u1[i1, i2] += v * b2d
                u2[i1, i2] += v * b2

            else:
                c0d,c1d,c2d,c3d,c4d = _B5d(s)
                c0, c1, c2, c3, c4  = _B5(s)
                # iw = -2,-1,0,1,2 (map manually for speed)
                v = din[i1 - 2, i2 + 1] - din[i1 + 2, i2]
                u1[i1, i2] += v * c0d; u2[i1, i2] += v * c0
                v = din[i1 - 1, i2 + 1] - din[i1 + 1, i2]
                u1[i1, i2] += v * c1d; u2[i1, i2] += v * c1
                v = din[i1 + 0, i2 + 1] - din[i1 + 0, i2]
                u1[i1, i2] += v * c2d; u2[i1, i2] += v * c2
                v = din[i1 + 1, i2 + 1] - din[i1 - 1, i2]
                u1[i1, i2] += v * c3d; u2[i1, i2] += v * c3
                v = din[i1 + 2, i2 + 1] - din[i1 - 2, i2]
                u1[i1, i2] += v * c4d; u2[i1, i2] += v * c4


# -------------------------------
# Public API (same signature + smoothing selector)
# -------------------------------
def pwddip2d(
    d: npt.ArrayLike,
    niter: int = 5,
    liter: int = 20,
    order: int = 2,
    nsmooth: Tuple[int, int] = (10, 10),
    damp: float = 0.0,
    smoothing: str = "triangle",  # <-- "triangle" or "boxcar"
) -> NDArray:
    
    r"""
    Estimate local slopes in 2D data using Plane-Wave Destruction (PWD).

    This implementation follows the plane-wave destruction method
    for local slope estimation (Claerbout, 1999; Fomel, 2002).
    Slopes :math:`\sigma(z,x)` are estimated by solving a local
    least-squares system at each iteration, with smoothing applied
    to stabilize the inversion.

    The inversion is preconditioned by a structure-aligned smoother,
    chosen as either a triangular kernel (two boxcar passes) or a
    single-pass boxcar. Iterative updates are accumulated into the
    final slope field.

    Parameters
    ----------
    d : :obj:`numpy.ndarray` or :obj:`array_like`
        Input 2D array of shape ``(nz, nx)`` containing seismic
        or image data.
    niter : :obj:`int`, optional
        Number of outer PWD iterations. Default is ``5``.
    liter : :obj:`int`, optional
        Maximum number of inner least-squares iterations
        (LSQR/LSMR). Default is ``20``.
    order : :obj:`int`, optional
        Accuracy order of the all-pass filter. Use ``1`` (3-tap)
        or ``2`` (5-tap). Default is ``2``.
    nsmooth : :obj:`tuple` of :obj:`int`, optional
        Smoothing lengths ``(ny, nx)``.
        - If ``smoothing="boxcar"``: window lengths of the moving average.
        - If ``smoothing="triangle"``: desired base lengths of the
          triangular kernel; realized base lengths are
          :math:`(2L_y-1, 2L_x-1)` with
          :math:`L_y=\lfloor(\text{ny}+1)/2\rfloor`,
          :math:`L_x=\lfloor(\text{nx}+1)/2\rfloor`.
        Default is ``(10, 10)``.
    damp : :obj:`float`, optional
        Damping factor for the least-squares solver.
        Default is ``0.0``.
    smoothing : :obj:`str`, optional
        Choice of smoothing preconditioner:
        - ``"triangle"``: two-pass triangular smoothing (default).
        - ``"boxcar"``: single-pass moving average.

    Returns
    -------
    sigma : :obj:`numpy.ndarray`
        Estimated slope field of shape ``(nz, nx)``.

    Notes
    -----
    The PWD method finds local slopes by minimizing the difference
    between data shifted along estimated dips and adjacent traces.
    A preconditioner enforces smoothness in the slope field.

    References
    ----------
    - Claerbout, J. F. (1992). ``Earth sounding Analysis Processing vs Inversion``.
    - Fomel, S. (2002). "Applications of plane-wave destruction filters."
      *Geophysics*, 67(6), 1946-1960.

    Examples
    --------
    >>> import numpy as np
    >>> from pylops.utils.seismicevents import linear2d
    >>> n1, n2 = 100, 50
    >>> t = np.linspace(0, 1, n1)
    >>> x = np.arange(n2)
    >>> data, _, _ = linear2d(t, x, t0=0.2, vel=1.0)
    >>> sigma = pwddip2d(data, niter=3, nsmooth=(5,5))
    >>> sigma.shape
    (100, 50)
    """
    din = np.ascontiguousarray(np.asarray(d, dtype=np.float32))
    nz, nx = din.shape

    sigma = np.zeros((nz, nx), dtype=np.float32)
    delta_sigma = np.zeros_like(sigma)

    u1 = np.zeros_like(sigma)
    u2 = np.zeros_like(sigma)

    # --- preconditioner choice ---
    if smoothing.lower() == "triangle":
        Sop = triangular_smoothing_from_boxcars(nsmooth=nsmooth, dims=(nz, nx), dtype="float32")
    elif smoothing.lower() == "boxcar":
        Sop = pylops.Smoothing2D(nsmooth=nsmooth, dims=(nz, nx), dtype="float32")
    else:
        raise ValueError("smoothing must be 'triangle' or 'boxcar'")

    for _ in range(niter):
        conv_allpass_numba(din, sigma, order, u1, u2)

        # Diagonal with model dims so PyLops keeps metadata
        Dop = pylops.Diagonal(u1.ravel().astype("float32"), dtype="float32")

        delta_sigma[:] = pylops.optimization.leastsquares.preconditioned_inversion(
            Dop,
            (-u2.ravel()).astype(np.float32, copy=False),
            Sop,
            damp=damp,
            iter_lim=liter,
            show=0,
        )[0].reshape(nz, nx)

        sigma += delta_sigma

    return sigma.astype(np.float32, copy=False)