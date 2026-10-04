# metrics.py: PSNR, SSIM, lesion contrast, background noise, contrast recovery.
import numpy as np
from skimage.metrics import structural_similarity


def psnr(truth: np.ndarray, rec: np.ndarray, mask: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) over mask pixels with data_range=1.0.

    Array shapes:
        truth: [n, n] float64 in [0, 1].
        rec: [n, n] float64 in [0, 1].
        mask: [n, n] bool.
        Returns: float (inf if MSE is 0.0).
    """
    diff = truth[mask] - rec[mask]
    mse = np.mean(diff ** 2)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def ssim(truth: np.ndarray, rec: np.ndarray, mask: np.ndarray) -> float:
    """Calculate Structural Similarity Index (SSIM) averaged over mask pixels.

    Array shapes:
        truth: [n, n] float64 in [0, 1].
        rec: [n, n] float64 in [0, 1].
        mask: [n, n] bool.
        Returns: float.
    """
    _, ssim_map = structural_similarity(
        truth,
        rec,
        win_size=7,
        data_range=1.0,
        gaussian_weights=False,
        full=True,
    )
    return float(np.mean(ssim_map[mask]))


def lesion_contrast(img: np.ndarray, lesion_roi: np.ndarray, background_roi: np.ndarray) -> float:
    """Calculate lesion contrast: (mu_L - mu_B) / mu_B.

    Array shapes:
        img: [n, n] float64.
        lesion_roi: [n, n] bool.
        background_roi: [n, n] bool.
        Returns: float.
    """
    mu_l = float(np.mean(img[lesion_roi]))
    mu_b = float(np.mean(img[background_roi]))
    if mu_b == 0.0:
        return 0.0
    return float((mu_l - mu_b) / mu_b)


def background_noise(img: np.ndarray, background_roi: np.ndarray) -> float:
    """Calculate background noise: sigma_B / mu_B with ddof=0.

    Array shapes:
        img: [n, n] float64.
        background_roi: [n, n] bool.
        Returns: float.
    """
    mu_b = float(np.mean(img[background_roi]))
    if mu_b == 0.0:
        return 0.0
    sigma_b = float(np.std(img[background_roi], ddof=0))
    return float(sigma_b / mu_b)


def contrast_recovery(contrast_rec: float, contrast_truth: float) -> float:
    """Calculate contrast recovery coefficient: contrast_rec / contrast_truth.

    Returns: float.
    """
    if contrast_truth == 0.0:
        return 0.0
    return float(contrast_rec / contrast_truth)


def score(
    truth: np.ndarray,
    rec_scaled: np.ndarray,
    mask: np.ndarray,
    lesion_roi: np.ndarray,
    background_roi: np.ndarray,
) -> dict[str, float]:
    """Score reconstruction against ground truth across all five metrics.

    Array shapes:
        truth: [n, n] float64 in [0, 1].
        rec_scaled: [n, n] float64 in [0, 1].
        mask: [n, n] bool.
        lesion_roi: [n, n] bool.
        background_roi: [n, n] bool.
        Returns: dict with keys psnr, ssim, lesion_contrast, background_noise, contrast_recovery.
    """
    c_truth = lesion_contrast(truth, lesion_roi, background_roi)
    c_rec = lesion_contrast(rec_scaled, lesion_roi, background_roi)

    return {
        "psnr": psnr(truth, rec_scaled, mask),
        "ssim": ssim(truth, rec_scaled, mask),
        "lesion_contrast": c_rec,
        "background_noise": background_noise(rec_scaled, background_roi),
        "contrast_recovery": contrast_recovery(c_rec, c_truth),
    }
