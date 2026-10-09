# methods.py: reconstruction methods fbp, mlem, osem, mlem_tv.
# Takes: a CountingOperator, sinogram m [n_angles, n_det], and method parameters.
# Returns: fbp -> image [n,n] (data units); iterative methods -> (x [n,n], history).
"""Reconstruction methods for tomographic imaging."""

from __future__ import annotations

import numpy as np

from projector import CountingOperator


def fbp(
    op: CountingOperator,
    m: np.ndarray,
    thetas_deg: np.ndarray,
    mask: np.ndarray,
    window: str = "hann",
) -> np.ndarray:
    """Filtered back-projection reconstruction.

    Parameters
    ----------
    op : CountingOperator
        Coarse projection operator wrapping sparse A of shape (n_angles*n_det, n*n).
    m : np.ndarray
        Sinogram [n_angles, n_det], float64.
    thetas_deg : np.ndarray
        Projection angles in degrees, shape (n_angles,).
    mask : np.ndarray
        Boolean inscribed-circle mask [n, n].
    window : str
        Filter window: "hann" or "none".

    Returns
    -------
    rec : np.ndarray
        Reconstructed image [n, n] in data units, zero outside mask, no clipping.
    """
    thetas_deg = np.atleast_1d(np.asarray(thetas_deg, dtype=np.float64))
    n_angles = len(thetas_deg)
    n_det = m.shape[1] if m.ndim == 2 else op.shape[0] // n_angles
    n_sq = op.shape[1]
    n = int(round(np.sqrt(n_sq)))

    # Reshape sinogram to 2-D if needed
    sino = m.reshape(n_angles, n_det).astype(np.float64)

    # Pad to 4 * n_det (D34)
    n_pad = 4 * n_det

    # Build ramp filter |omega| in frequency domain
    freqs = np.fft.fftfreq(n_pad, d=1.0)  # cycles per detector bin
    ramp = np.abs(freqs)

    # Apply window
    if window == "hann":
        # Hann window: 0.5 * (1 + cos(2*pi*omega))
        # This is applied as a multiplicative apodisation to the ramp filter.
        # At omega=0 -> 1.0; at omega=0.5 (Nyquist) -> 0.0
        hann = 0.5 * (1.0 + np.cos(2.0 * np.pi * freqs))
        filt = ramp * hann
    elif window == "none":
        filt = ramp
    else:
        raise ValueError(f"Unknown window '{window}', use 'hann' or 'none'")

    # Filter each projection: pad, FFT, multiply by filter, IFFT, crop
    filtered = np.zeros_like(sino)
    for i in range(n_angles):
        padded = np.zeros(n_pad, dtype=np.float64)
        padded[:n_det] = sino[i, :]
        ft = np.fft.fft(padded)
        ft *= filt
        filtered[i, :] = np.real(np.fft.ifft(ft))[:n_det]

    # Back-project: flatten filtered sinogram, apply A^T
    bp = op.back(filtered.ravel())
    rec = bp.reshape(n, n)

    # Scale by pi / n_angles (D33 analytical derivation)
    rec *= np.pi / n_angles

    # Zero outside mask
    rec[~mask] = 0.0

    return rec


def mlem(
    op: CountingOperator,
    m: np.ndarray,
    n_iter: int,
    mask: np.ndarray,
    callback=None,
) -> tuple[np.ndarray, list[dict]]:
    """Maximum Likelihood Expectation Maximization (MLEM) reconstruction.

    Parameters
    ----------
    op : CountingOperator
        Coarse projection operator wrapping sparse A.
    m : np.ndarray
        Sinogram [n_angles, n_det], float64.
    n_iter : int
        Number of iterations.
    mask : np.ndarray
        Boolean inscribed-circle mask [n, n].
    callback : callable, optional
        Callback function called after each iteration as callback(k, x, history[-1]).

    Returns
    -------
    x : np.ndarray
        Reconstructed image [n, n].
    history : list[dict]
        List of iteration history dicts containing "iter" and "loglik".
    """
    n_sq = op.shape[1]
    n = int(round(np.sqrt(n_sq)))
    m_flat = m.ravel().astype(np.float64)

    # Flat start
    x = mask.astype(np.float64) * float(mask.sum()) / float(n_sq)

    # Sensitivity image
    s = op.back(np.ones_like(m_flat))
    s = np.maximum(s, 1e-6)
    s = s.reshape(n, n)
    s[~mask] = 0.0

    history = []
    
    eps = 1e-9

    for k in range(n_iter):
        ax = op.forward(x.ravel())
        
        # Log-likelihood
        ax_eps = ax + eps
        loglik = float(np.sum(m_flat * np.log(ax_eps) - ax_eps))
        history.append({"iter": k, "loglik": loglik})
        
        # Update
        ratio = m_flat / np.maximum(ax, eps)
        back_ratio = op.back(ratio).reshape(n, n)
        
        # MUTATION 1: remove /s step (no sensitivity correction)
        # MUTATION 2: use additive update x = x + ratio instead of multiplicative
        with np.errstate(divide="ignore", invalid="ignore"):
            x = x / s * back_ratio
        
        # MUTATION 3: skip zeroing outside mask
        x[~mask] = 0.0

        
        if callback is not None:
            callback(k, x, history[-1])

    return x, history
