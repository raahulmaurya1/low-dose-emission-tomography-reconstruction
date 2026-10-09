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
