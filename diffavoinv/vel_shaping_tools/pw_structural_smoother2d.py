import numpy as np
from pylops import LinearOperator


try:
    from numba import njit, prange
except ImportError:
    def njit(*args, **kwargs):
        def wrap(f): return f
        return wrap
    def prange(*args, **kwargs):
        return range(*args, **kwargs)

@njit(cache=True, fastmath=True, parallel=True)
def _spray_forward(m, sigma, radius, alpha, out):
    """
    Forward spray: for each (z,x) we walk +/-x up to `radius` steps, following slope.
    Bilinear in z when landing between rows. 'Nearest' clamping at edges.
    out[...] is incremented (+=) — NOT reset here.
    """
    nz, nx = m.shape
    for z0 in prange(nz):
        for x0 in range(nx):
            v0 = m[z0, x0]
            if v0 == 0.0:
                continue
            # center contribution
            out[z0, x0] += v0

            # +x direction
            amp = v0
            z = float(z0)
            for k in range(1, radius+1):
                x = x0 + k
                if x >= nx: break
                # slope at current integer lattice
                s = sigma[z0, min(x-1, nx-1)]
                z += s  # move along slope (dz/dx = sigma)
                amp *= alpha
                # bilinear on z (x is integer)
                zi = int(np.floor(z))
                t = z - zi
                zi0 = 0 if zi < 0 else (nz-1 if zi >= nz else zi)
                zi1 = 0 if zi+1 < 0 else (nz-1 if zi+1 >= nz else zi+1)
                out[zi0, x] += (1.0 - t) * amp
                out[zi1, x] += t * amp

            # -x direction
            amp = v0
            z = float(z0)
            for k in range(1, radius+1):
                x = x0 - k
                if x < 0: break
                s = sigma[z0, x]  # use slope at this column
                z -= s
                amp *= alpha
                zi = int(np.floor(z))
                t = z - zi
                zi0 = 0 if zi < 0 else (nz-1 if zi >= nz else zi)
                zi1 = 0 if zi+1 < 0 else (nz-1 if zi+1 >= nz else zi+1)
                out[zi0, x] += (1.0 - t) * amp
                out[zi1, x] += t * amp

@njit(cache=True, fastmath=True, parallel=True)
def _spray_adjoint(d, sigma, radius, alpha, out):
    """
    Adjoint gather: transpose of the forward.
    For each (z0,x0), we pull contributions from +/-x rays that would have
    landed at (z0,x0) in the forward pass, using the same bilinear weights.
    """
    nz, nx = d.shape
    for z0 in prange(nz):
        for x0 in range(nx):
            acc = d[z0, x0]  # center (self) term
            # +x contributions coming back from the right
            amp = 1.0
            z = float(z0)
            for k in range(1, radius+1):
                x = x0 + k
                if x >= nx: break
                s = sigma[z0, min(x-1, nx-1)]
                z += s
                amp *= alpha
                zi = int(np.floor(z))
                t = z - zi
                zi0 = 0 if zi < 0 else (nz-1 if zi >= nz else zi)
                zi1 = 0 if zi+1 < 0 else (nz-1 if zi+1 >= nz else zi+1)
                # in forward we wrote to (zi0,x) and (zi1,x)
                acc += (1.0 - t) * amp * d[zi0, x]
                acc += t * amp * d[zi1, x]

            # -x contributions coming back from the left
            amp = 1.0
            z = float(z0)
            for k in range(1, radius+1):
                x = x0 - k
                if x < 0: break
                s = sigma[z0, x]
                z -= s
                amp *= alpha
                zi = int(np.floor(z))
                t = z - zi
                zi0 = 0 if zi < 0 else (nz-1 if zi >= nz else zi)
                zi1 = 0 if zi+1 < 0 else (nz-1 if zi+1 >= nz else zi+1)
                acc += (1.0 - t) * amp * d[zi0, x]
                acc += t * amp * d[zi1, x]

            out[z0, x0] += acc

class PWSprayer2D(LinearOperator):
    
    r"""
    Plane-Wave Sprayer in 2D.

    Forward mode sprays (paints) each input value along local structural
    slopes in the :math:`\pm x` directions with exponential decay.
    The adjoint mode gathers contributions back along the same slope
    trajectories. Together, these define a linear operator that propagates
    information preferentially along structural dips.

    Parameters
    ----------
    dims : :obj:`tuple` of :obj:`int`
        Dimensions of the 2D model (``nz, nx``).
    sigma : :obj:`numpy.ndarray`
        Local slope field of shape ``(nz, nx)``, in samples per trace
        (:math:`dz/dx`).
    radius : :obj:`int`, optional
        Maximum number of steps along each :math:`\pm x` direction to spray
        or gather. Controls the spatial extent of spreading. Default is ``8``.
    alpha : :obj:`float`, optional
        Geometric decay factor per step (:math:`0 < \alpha \leq 1`).
        Higher values propagate energy farther. Default is ``0.9``.
    dtype : :obj:`str` or :obj:`numpy.dtype`, optional
        Data type of the operator. Default is ``'float32'``.

    Notes
    -----
    - The forward operator distributes each sample along rays following
      local slope :math:`\sigma(z,x)`, using bilinear interpolation in depth.
    - The adjoint operator gathers contributions back from the same rays,
      making this a true linear operator pair.
    - Effective smoothing along dip grows with larger ``radius`` and
      higher ``alpha``.

    See Also
    --------
    PWSmoother2D : Structure-aligned smoother (``Sprayer.T @ Sprayer``)
    pwddip2d : Local slope estimation using plane-wave destruction

    Examples
    --------
    >>> import numpy as np
    >>> from pylops.utils import dottest
    >>> nz, nx = 40, 30
    >>> sigma = np.zeros((nz, nx), dtype='float32')  # flat slope
    >>> P = PWSprayer2D(dims=(nz, nx), sigma=sigma, radius=4, alpha=0.9)
    >>> x = np.zeros(nz*nx, dtype='float32')
    >>> x[nz//2 * nx + nx//2] = 1.0   # impulse in the center
    >>> y = P @ x                     # spray along horizontal direction
    >>> dottest(P, nz*nx, nz*nx, complexflag=False)
    True
    """

    def __init__(self, dims, sigma, radius=8, alpha=0.9, dtype="float32"):
        assert len(dims) == 2
        nz, nx = dims
        self.nz, self.nx = int(nz), int(nx)
        self.shape = (nz*nx, nz*nx)
        self.dtype = np.dtype(dtype)
        self.radius = int(radius)
        self.alpha = float(alpha)
        # store slope as float32 contiguous
        self.sigma = np.ascontiguousarray(sigma.astype(np.float32))
        super().__init__(dtype=self.dtype)

    def _matvec(self, x):
        x2 = np.asarray(x, dtype=np.float32, order="C").reshape(self.nz, self.nx)
        y2 = np.zeros_like(x2)
        _spray_forward(x2, self.sigma, self.radius, self.alpha, y2)
        return y2.ravel().astype(self.dtype, copy=False)

    def _rmatvec(self, x):
        x2 = np.asarray(x, dtype=np.float32, order="C").reshape(self.nz, self.nx)
        y2 = np.zeros_like(x2)
        _spray_adjoint(x2, self.sigma, self.radius, self.alpha, y2)
        return y2.ravel().astype(self.dtype, copy=False)



class PWSmoother2D(LinearOperator):
    
    r"""
    Structure-aligned 2D smoother based on plane-wave spraying.

    This operator builds a symmetric, positive semi-definite (PSD) smoother
    aligned with local structural dips. It is defined as

    .. math::

        S = P^\top P

    where :math:`P` is a :class:`PWSprayer2D` operator that propagates
    values along local slopes. The composition ``P.T @ P`` produces a
    correlation-like operator that smooths preferentially along dip
    directions.

    The resulting operator can be used as a regularizer or preconditioner
    in inverse problems to enforce structural smoothness.

    Parameters
    ----------
    dims : :obj:`tuple` of :obj:`int`
        Dimensions of the 2D model (``nz, nx``).
    sigma : :obj:`numpy.ndarray`
        Local slope field of shape ``(nz, nx)``, in samples per trace
        (:math:`dz/dx`).
    radius : :obj:`int`, optional
        Maximum number of steps (in samples) to spray along ``±x``.
        Default is ``8``.
    alpha : :obj:`float`, optional
        Geometric decay factor per step (:math:`0<\alpha\leq 1`).
        Controls effective smoothing length. Default is ``0.9``.
    dtype : :obj:`str` or :obj:`numpy.dtype`, optional
        Data type of the operator. Default is ``'float32'``.

    Notes
    -----
    - The smoother is symmetric by construction.
    - Effective correlation length along dip increases with larger
      ``radius`` and higher ``alpha``.
    - Across-dip coupling is minimal (only through interpolation and dip
      variability).

    See Also
    --------
    PWSprayer2D : Forward sprayer/gather operator
    pwddip2d : Local slope estimation using plane-wave destruction

    Examples
    --------
    >>> import numpy as np
    >>> from pylops.utils import dottest
    >>> nz, nx = 50, 30
    >>> sigma = np.zeros((nz, nx), dtype='float32')  # flat structure
    >>> Sop = PWSmoother2D(dims=(nz, nx), sigma=sigma, radius=4, alpha=0.9)
    >>> x = np.random.randn(nz*nx).astype('float32')
    >>> y = Sop @ x
    >>> dottest(Sop, nz*nx, nz*nx, complexflag=False)
    True
    """
    def __init__(self, dims, sigma, radius=8, alpha=0.9, dtype="float32"):
        self.nz, self.nx = dims
        self.shape = (self.nz*self.nx, self.nz*self.nx)
        self.dtype = np.dtype(dtype)
        self._sprayer = PWSprayer2D(dims=dims, sigma=sigma, radius=radius, alpha=alpha, dtype=dtype)
        super().__init__(dtype=self.dtype)

    def _matvec(self, x):
        # y = Spray^T (Spray x)
        y = self._sprayer @ x
        y = self._sprayer.H @ y
        return y

    def _rmatvec(self, x):
        # symmetric
        return self._matvec(x)