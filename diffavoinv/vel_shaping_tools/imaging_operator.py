import numpy as np
import pylops


# ---------------------------------------------------------
# Helper: nonlinear forward (for simulation, not an operator)
# ---------------------------------------------------------
def forward_F_nonlinear(Dz: pylops.LinearOperator,
                        v: np.ndarray,
                        v0: np.ndarray,
                        eps_inv: float = 1e-6,
                        order: str = "C",
                        dtype=np.float32) -> np.ndarray:
    """
    Compute d = Dz [ g(v) - g(v0) ], where g(u)=1/((u+2)^2 + eps_inv).
    Shapes: v, v0 -> (nz, nx).  Output is a data vector in Dz's range.
    """
    assert v.shape == v0.shape, "v and v0 must have same (nz, nx) shape"
    nz, nx = v.shape
    N = nz * nx

    v  = v.astype(dtype, copy=False)
    v0 = v0.astype(dtype, copy=False)

    def g(u):
        return 1.0 / ((u + 2.0)**2 + eps_inv)

    gv  = g(v ).reshape(N, order=order)
    gv0 = g(v0).reshape(N, order=order)

    q = gv - gv0
    d = Dz @ q
    return np.asarray(d, dtype=dtype)


# ----------------------------------------------------------------
# Linearized operator: J(v_ref) δv = Dz [ g'(v_ref) ⊙ δv ]
# ----------------------------------------------------------------
class ImageJacobian(pylops.LinearOperator):
    """
    PyLops LinearOperator for the linearization of F(v)=Dz[g(v)-g(v0)]
    around a reference model v_ref:

        J(v_ref) δv = Dz [ g'(v_ref) ⊙ δv ],

    where g(u) = 1 / ((u+2)^2 + eps_inv) and
          g'(u) = -2 (u+2) / (( (u+2)^2 + eps_inv )^2).

    Parameters
    ----------
    Dz : pylops.LinearOperator
        Operator acting on vectorized (nz*nx,) model space (e.g., FirstDerivative in z).
    v_ref : np.ndarray, shape (nz, nx)
        Reference model at which we linearize.
    eps_inv : float, optional
        Small positive constant in g(u). Default 1e-6.
    order : {"C","F"}, optional
        Memory order for vectorization. Default "C".
    dtype : numpy dtype, optional
        Storage/compute dtype. Default np.float32.
    """
    def __init__(self,
                 Dz: pylops.LinearOperator,
                 v_ref: np.ndarray,
                 eps_inv: float = 1e-6,
                 order: str = "C",
                 dtype=np.float32):
        assert v_ref.ndim == 2, "v_ref must be (nz, nx)"
        self.Dz = Dz
        self.nz, self.nx = v_ref.shape
        self.N = self.nz * self.nx
        self.eps_inv = float(eps_inv)
        self.order = order
        self.dtype = dtype

        # Cache g'(v_ref) as a vector multiplier
        self._set_point(v_ref)

        # Initialize parent LinearOperator
        super().__init__(dtype=self.dtype,
                         shape=(Dz.shape[0], self.N))

    # ----- public API to update linearization point -----
    def set_point(self, v_ref: np.ndarray):
        """Update reference model and refresh the Jacobian diagonal."""
        self._set_point(v_ref)

    # ----- internals -----
    def _gprime(self, u: np.ndarray) -> np.ndarray:
        den = ((u + 2.0) ** 2 + self.eps_inv)
        return (-2.0 * (u + 2.0)) / (den ** 2)

    def _set_point(self, v_ref: np.ndarray):
        assert v_ref.shape == (self.nz, self.nx)
        v_ref = v_ref.astype(self.dtype, copy=False)
        gp = self._gprime(v_ref).reshape(self.N, order=self.order)
        self.gp_vec = gp  # (N,)

    # ----- LinearOperator core: forward and adjoint -----
    def _matvec(self, x: np.ndarray) -> np.ndarray:
        """
        y = Dz [ g'(v_ref) ⊙ x ]
        """
        x = np.asarray(x, dtype=self.dtype).reshape(self.N, order=self.order)
        q = self.gp_vec * x                     # (N,)
        y = self.Dz @ q                         # range of Dz
        return np.asarray(y, dtype=self.dtype)

    def _rmatvec(self, y: np.ndarray) -> np.ndarray:
        """
        x = [ g'(v_ref) ⊙ (Dz^H y) ]
        """
        y = np.asarray(y, dtype=self.dtype)
        z = self.Dz.H @ y                       # (N,)
        x = self.gp_vec * z
        return np.asarray(x, dtype=self.dtype)
