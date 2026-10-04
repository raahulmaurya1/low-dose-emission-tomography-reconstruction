# phantom.py: test images and ROIs.
import numpy as np
from scipy.ndimage import binary_erosion
from skimage.data import shepp_logan_phantom
from skimage.transform import resize


def _coords_fine(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return fine-grid coordinates (YY, XX) centered at image origin with coarse pixel spacing 1."""
    yf = (np.arange(2 * n) - (2 * n - 1) / 2.0) / 2.0
    return np.meshgrid(yf, yf, indexing="ij")


def make_fine_phantom(n: int, lesion: bool = False, lesion_contrast: float = 0.25) -> np.ndarray:
    """Generate fine-grid Shepp-Logan phantom on a 2n x 2n grid in [0, 1].

    Parameters:
        n: Coarse image dimension (multiple of 32).
        lesion: Whether to add synthetic lesion disk.
        lesion_contrast: Relative contrast (mu_L - mu_B)/mu_B over 0.20 tissue, default 0.25.

    Array shapes:
        Returns: fine_truth [2n, 2n] float64.
    """
    if n % 32 != 0:
        raise ValueError(f"Image dimension n must be a multiple of 32, got {n}.")

    cf = (2 * n - 1) / 2.0
    yy_idx, xx_idx = np.mgrid[:2 * n, :2 * n]
    fine_mask = (yy_idx - cf) ** 2 + (xx_idx - cf) ** 2 <= n ** 2

    base = resize(shepp_logan_phantom(), (2 * n, 2 * n), anti_aliasing=True) * fine_mask
    if not lesion:
        return base.astype(np.float64)

    yy, xx = _coords_fine(n)
    cy_l, cx_l = 4.0 * n / 32.0, 7.0 * n / 32.0
    r = 3.0 * n / 64.0
    d_l = (yy - cy_l) ** 2 + (xx - cx_l) ** 2 <= r ** 2

    fine = base.copy()
    fine[d_l] = 0.20 * (1.0 + lesion_contrast)
    return fine.astype(np.float64)


def block_mean(fine: np.ndarray) -> np.ndarray:
    """Compute 2x2 block mean downsampling from 2n x 2n to n x n.

    Array shapes:
        fine: [2n, 2n] float64.
        Returns: coarse [n, n] float64.
    """
    h, w = fine.shape
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError(f"Input dimensions must be even, got shape {fine.shape}.")
    return fine.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def make_phantom(n: int, lesion: bool = False, lesion_contrast: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Generate coarse truth and inscribed circular mask.

    Array shapes:
        Returns: (truth [n, n] float64, mask [n, n] bool).
    """
    fine = make_fine_phantom(n, lesion=lesion, lesion_contrast=lesion_contrast)
    truth = block_mean(fine)

    yy, xx = np.mgrid[:n, :n]
    c = (n - 1) / 2.0
    mask = (yy - c) ** 2 + (xx - c) ** 2 <= (n / 2.0) ** 2
    return truth * mask, mask


def lesion_rois(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate boolean masks for lesion ROI and matching background ROI.

    Array shapes:
        Returns: (lesion_roi [n, n] bool, background_roi [n, n] bool).
    """
    if n % 32 != 0:
        raise ValueError(f"Image dimension n must be a multiple of 32, got {n}.")

    yy, xx = _coords_fine(n)
    cy_l, cx_l = 4.0 * n / 32.0, 7.0 * n / 32.0
    cy_b, cx_b = 33.0 * n / 128.0, -20.0 * n / 128.0
    r_l = 3.0 * n / 64.0
    r_b = 8.5 * n / 128.0

    d_l = (yy - cy_l) ** 2 + (xx - cx_l) ** 2 <= r_l ** 2
    d_b = (yy - cy_b) ** 2 + (xx - cx_b) ** 2 <= r_b ** 2

    coarse_l = d_l.reshape(n, 2, n, 2).all(axis=(1, 3))
    coarse_b = d_b.reshape(n, 2, n, 2).all(axis=(1, 3))

    lesion_roi = binary_erosion(coarse_l, iterations=1)
    background_roi = binary_erosion(coarse_b, iterations=1)
    return lesion_roi.astype(bool), background_roi.astype(bool)
