# scan.py: scan simulation, fine-grid projection + Poisson noise.
"""Simulates low-dose tomographic projection data using fine-grid forward projection.

Prevents inverse crime by projecting on a 2x fine grid with fine detector bins
before downsampling detector bins in pairs.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from projector import build_operator


def simulate_scan(
    fine_truth: np.ndarray,
    thetas_deg: np.ndarray | list[float] | float,
    counts: float,
    seed: int,
    A_fine: csr_matrix | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Simulate tomographic scan with fine-grid projection and Poisson noise.

    Parameters
    ----------
    fine_truth : np.ndarray
        Ground truth image on fine grid [2n, 2n] in [0, 1].
    thetas_deg : np.ndarray | list[float] | float
        Projection angles in degrees.
    counts : float
        Target total expected counts across all detector bins and angles.
    seed : int
        RNG seed for reproducibility.
    A_fine : csr_matrix | None, optional
        Precomputed fine operator of shape (n_angles * 2 * (n + 2), 4 * n * n).
        If None, built on the fly using build_operator(2*n, thetas_deg, n_det=2*(n+2)).

    Returns
    -------
    m : np.ndarray
        Noisy Poisson sinogram [n_angles, n + 2], integer-valued float64 counts.
    scale : float
        Calibration scale factor = counts / sum(expected).
    expected : np.ndarray
        Noise-free expected projection profile [n_angles, n + 2].
    """
    fine_truth = np.asarray(fine_truth, dtype=np.float64)
    if fine_truth.ndim != 2 or fine_truth.shape[0] != fine_truth.shape[1] or fine_truth.shape[0] % 2 != 0:
        raise ValueError("fine_truth must be a square 2D array with even side length (2n x 2n)")
    if counts <= 0:
        raise ValueError("counts must be positive")

    n = fine_truth.shape[0] // 2
    thetas_deg_arr = np.atleast_1d(thetas_deg)
    n_angles = len(thetas_deg_arr)

    if A_fine is None:
        A_fine = build_operator(2 * n, thetas_deg_arr, n_det=2 * (n + 2))

    fine_activity = fine_truth / 4.0
    fine_proj = A_fine @ fine_activity.ravel()
    fine_proj_2d = fine_proj.reshape(n_angles, 2 * (n + 2))
    expected = fine_proj_2d.reshape(n_angles, n + 2, 2).sum(axis=2)

    scale = float(counts / expected.sum())
    rng = np.random.default_rng(seed)
    m = rng.poisson(scale * expected).astype(np.float64)

    return m, scale, expected
