# projector.py: operator A / A^T and the call-counting wrapper.
"""Discrete Radon projection operator and call-counting wrapper."""

from typing import Optional, Sequence, Tuple, Union
import numpy as np
from scipy.sparse import csr_matrix, vstack


def _angle_block(x: np.ndarray, y: np.ndarray, theta: float, n_det: int) -> csr_matrix:
    """One angle of A: csr_matrix [n_det, n*n]. x, y: pixel-centre coordinates, shape (n*n,)."""
    n_pixels = x.size
    c_det = (n_det - 1) / 2.0
    u = x * np.cos(theta) + y * np.sin(theta) + c_det  # s = x cos + y sin, shifted to bin index
    b0 = np.floor(u).astype(np.int64)
    w1 = u - b0
    bins = np.concatenate([b0, b0 + 1])
    vals = np.concatenate([1.0 - w1, w1])
    pix = np.tile(np.arange(n_pixels, dtype=np.int64), 2)
    keep = (bins >= 0) & (bins < n_det)  # drop weights that fall outside the detector
    return csr_matrix((vals[keep], (bins[keep], pix[keep])), shape=(n_det, n_pixels), dtype=np.float64)


def build_operator(
    n: int,
    thetas_deg: Union[float, Sequence[float], np.ndarray],
    n_det: Optional[int] = None,
) -> csr_matrix:
    """Build sparse Radon forward projection operator A.

    Pixel-driven, linear interpolation. Pixel centres at (i - (n-1)/2); x = column offset,
    y = row offset (downwards); s = x cos(theta) + y sin(theta); detector bins centred, spacing 1.
    Vectorised over pixels; one block per angle to bound peak memory.

    Parameters
    ----------
    n : image dimension (n x n image).
    thetas_deg : scalar or sequence of shape (n_angles,), projection angles in degrees.
    n_det : number of detector bins; None means n + 2 (guard bins).

    Returns
    -------
    A : csr_matrix of shape (n_angles * n_det, n * n), float64, angle-major rows,
        explicit zeros removed, sorted indices.
    """
    if n_det is None:
        n_det = n + 2
    thetas_rad = np.deg2rad(np.atleast_1d(np.asarray(thetas_deg, dtype=np.float64)))
    c_img = (n - 1) / 2.0
    yy, xx = np.mgrid[:n, :n]
    x = (xx - c_img).ravel().astype(np.float64)
    y = (yy - c_img).ravel().astype(np.float64)

    A = vstack([_angle_block(x, y, t, n_det) for t in thetas_rad], format="csr")
    A = csr_matrix(A, dtype=np.float64)
    A.eliminate_zeros()
    A.sort_indices()
    return A


class CountingOperator:
    """Wrapper around sparse projection operator A tracking forward and back calls."""

    def __init__(self, A: csr_matrix) -> None:
        self.A: csr_matrix = A
        self.AT: csr_matrix = A.T.tocsr()
        self.shape: Tuple[int, int] = A.shape
        self.n_forward: int = 0
        self.n_back: int = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward projection y = A x. Input x shape: (n*n,), returns (n_angles*n_det,)."""
        if np.shape(x) != (self.shape[1],):
            raise ValueError(f"forward expects shape ({self.shape[1]},), got {np.shape(x)}")
        self.n_forward += 1
        return self.A @ x

    def back(self, y: np.ndarray) -> np.ndarray:
        """Back projection x = A^T y. Input y shape: (n_angles*n_det,), returns (n*n,)."""
        if np.shape(y) != (self.shape[0],):
            raise ValueError(f"back expects shape ({self.shape[0]},), got {np.shape(y)}")
        self.n_back += 1
        return self.AT @ y
