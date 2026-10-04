# test_basics.py: phantom, metrics, and scan tests (Steps 1 and 3).
import numpy as np
import pytest

from config import Settings
from phantom import make_fine_phantom, block_mean, make_phantom, lesion_rois
from metrics import psnr, ssim, lesion_contrast, background_noise, contrast_recovery, score


def test_import_config_defaults():
    """Verify pytest.ini allows importing config and Settings has correct defaults."""
    s = Settings()
    assert s.n == 128
    assert s.n_angles == 60
    assert s.counts == 2e5
    assert s.seed == 0
    assert s.n_iter == 100
    assert s.n_subsets == 1
    assert s.beta == 0.0
    assert s.lesion is False


@pytest.mark.parametrize("n", [64, 128])
def test_phantom_shape_range_mask(n):
    """Test phantom shape, value range, inscribed circular mask, and zero support outside."""
    truth, mask = make_phantom(n, lesion=False)
    assert truth.shape == (n, n)
    assert mask.shape == (n, n)
    assert mask.dtype == bool
    assert np.all(truth >= 0.0)
    assert np.all(truth <= 1.0)
    assert np.all(truth[~mask] == 0.0)

    yy, xx = np.mgrid[:n, :n]
    c = (n - 1) / 2.0
    expected_mask = (yy - c) ** 2 + (xx - c) ** 2 <= (n / 2.0) ** 2
    np.testing.assert_array_equal(mask, expected_mask)


@pytest.mark.parametrize("n", [64, 128])
def test_fine_phantom_and_block_mean(n):
    """Test that make_phantom equals block_mean(make_fine_phantom) and block_mean preserves mean."""
    fine_clean = make_fine_phantom(n, lesion=False)
    assert fine_clean.shape == (2 * n, 2 * n)
    assert np.all(fine_clean >= 0.0) and np.all(fine_clean <= 1.0)

    coarse_clean, mask = make_phantom(n, lesion=False)
    np.testing.assert_allclose(coarse_clean, block_mean(fine_clean), atol=1e-14)

    rng = np.random.default_rng(42)
    rand_fine = rng.random((2 * n, 2 * n))
    coarse_rand = block_mean(rand_fine)
    assert coarse_rand.shape == (n, n)
    np.testing.assert_allclose(rand_fine.mean(), coarse_rand.mean(), atol=1e-14)

    fine_lesion = make_fine_phantom(n, lesion=True)
    coarse_lesion, _ = make_phantom(n, lesion=True)
    np.testing.assert_allclose(coarse_lesion, block_mean(fine_lesion), atol=1e-14)


@pytest.mark.parametrize("n", [64, 128])
def test_lesion_and_background_rois(n):
    """Test ROI overlap, mask containment, pixel counts, and uniformity in truth."""
    truth_clean, mask = make_phantom(n, lesion=False)
    truth_lesion, _ = make_phantom(n, lesion=True)
    lesion_roi, background_roi = lesion_rois(n)

    assert lesion_roi.shape == (n, n)
    assert background_roi.shape == (n, n)
    assert lesion_roi.dtype == bool
    assert background_roi.dtype == bool

    # Pixel count checks: background ROI >= 150 at n=128
    if n == 128:
        assert background_roi.sum() >= 150
        assert lesion_roi.sum() == 68
    elif n == 64:
        assert background_roi.sum() >= 25
        assert lesion_roi.sum() == 12

    # No overlap
    assert not np.any(lesion_roi & background_roi)

    # Contained inside mask
    assert not np.any(lesion_roi & ~mask)
    assert not np.any(background_roi & ~mask)

    # Gap between lesion ROI and background ROI boundaries: >= 8 px at n=128 (>= 4 px at n=64)
    from scipy.ndimage import distance_transform_edt
    dist_to_lesion = distance_transform_edt(~lesion_roi)
    roi_gap = dist_to_lesion[background_roi].min()
    assert roi_gap >= (8.0 if n == 128 else 4.0)

    # Every ROI >= 5 px from any intensity edge in clean truth at n=128 (>= 2.5 px at n=64)
    edge_map = np.abs(truth_clean - 0.20) > 1e-6
    dist_to_edge = distance_transform_edt(~edge_map)
    assert dist_to_edge[lesion_roi].min() >= (5.0 if n == 128 else 2.5)
    assert dist_to_edge[background_roi].min() >= (5.0 if n == 128 else 2.5)

    # Background ROI uniformity in truth
    bg_vals = truth_clean[background_roi]
    assert np.ptp(bg_vals) < 1e-12
    assert np.std(bg_vals) < 1e-12
    np.testing.assert_allclose(np.mean(bg_vals), 0.20, atol=1e-6)

    # Lesion surroundings (clean truth at lesion ROI) uniformity
    clean_lesion_area = truth_clean[lesion_roi]
    assert np.ptp(clean_lesion_area) < 1e-12
    assert np.std(clean_lesion_area) < 1e-12
    np.testing.assert_allclose(np.mean(clean_lesion_area), 0.20, atol=1e-6)

    # Default lesion contrast +25% (intensity 0.25 over 0.20 background)
    lesion_vals = truth_lesion[lesion_roi]
    assert np.ptp(lesion_vals) < 1e-12
    assert np.std(lesion_vals) < 1e-12
    np.testing.assert_allclose(np.mean(lesion_vals), 0.25, atol=1e-6)
    expected_contrast = (0.25 - 0.20) / 0.20  # 0.25
    c_measured = lesion_contrast(truth_lesion, lesion_roi, background_roi)
    np.testing.assert_allclose(c_measured, expected_contrast, atol=1e-12)


def test_lesion_contrast_parameter():
    """Test custom lesion_contrast parameter in phantom generation."""
    n = 64
    truth_50, _ = make_phantom(n, lesion=True, lesion_contrast=0.50)
    l_roi, bg_roi = lesion_rois(n)
    # Intensity should be 0.20 * 1.50 = 0.30
    np.testing.assert_allclose(truth_50[l_roi].mean(), 0.30, atol=1e-6)
    np.testing.assert_allclose(lesion_contrast(truth_50, l_roi, bg_roi), 0.50, atol=1e-12)


def test_psnr_metrics():
    """Test PSNR of identical images is infinite, matches MSE, and fails on unmasked pixels."""
    n = 64
    _, mask = make_phantom(n)
    img = np.ones((n, n), dtype=np.float64) * 0.5
    img[~mask] = 0.0

    assert psnr(img, img, mask) == float("inf")

    # Hand-computed MSE case
    # delta = 0.1 on mask -> MSE = 0.01, data_range = 1.0 -> PSNR = 20.0 dB
    noisy = img.copy()
    noisy[mask] += 0.1
    np.testing.assert_allclose(psnr(img, noisy, mask), 20.0, atol=1e-12)

    # Perturbation outside mask must NOT affect PSNR
    noisy_outside = img.copy()
    noisy_outside[~mask] += 0.8
    assert psnr(img, noisy_outside, mask) == float("inf")


def test_ssim_metric():
    """Test SSIM with D10 settings: identical images yield 1.0."""
    n = 64
    truth, mask = make_phantom(n)
    val = ssim(truth, truth, mask)
    np.testing.assert_allclose(val, 1.0, atol=1e-12)


def test_contrast_and_noise_hand_computed():
    """Test contrast, background noise, and contrast recovery on toy arrays."""
    n = 10
    mask = np.ones((n, n), dtype=bool)
    lesion_roi = np.zeros((n, n), dtype=bool)
    background_roi = np.zeros((n, n), dtype=bool)

    lesion_roi[1:3, 1:3] = True
    background_roi[5:7, 5:7] = True

    # Toy truth: mu_L = 0.25, mu_B = 0.20 -> contrast = 0.25
    toy_truth = np.zeros((n, n), dtype=np.float64)
    toy_truth[background_roi] = 0.20
    toy_truth[lesion_roi] = 0.25

    c_truth = lesion_contrast(toy_truth, lesion_roi, background_roi)
    np.testing.assert_allclose(c_truth, 0.25, atol=1e-12)

    # If swapped, contrast = (0.20 - 0.25)/0.25 = -0.20 != 0.25
    assert lesion_contrast(toy_truth, background_roi, lesion_roi) < 0.0

    b_noise = background_noise(toy_truth, background_roi)
    np.testing.assert_allclose(b_noise, 0.0, atol=1e-12)

    toy_rec = np.zeros((n, n), dtype=np.float64)
    toy_rec[background_roi] = np.array([0.18, 0.22, 0.19, 0.21])
    toy_rec[lesion_roi] = 0.24

    # c_rec = (0.24 - 0.20)/0.20 = 0.04/0.20 = 0.20
    c_rec = lesion_contrast(toy_rec, lesion_roi, background_roi)
    np.testing.assert_allclose(c_rec, 0.20, atol=1e-12)

    # contrast_recovery = 0.20 / 0.25 = 0.80
    recov = contrast_recovery(c_rec, c_truth)
    np.testing.assert_allclose(recov, 0.80, atol=1e-12)

    expected_std = np.std(np.array([0.18, 0.22, 0.19, 0.21]), ddof=0)
    expected_noise = expected_std / 0.20
    np.testing.assert_allclose(background_noise(toy_rec, background_roi), expected_noise, atol=1e-12)


def test_score_dict():
    """Test score function returns all expected keys and correct values."""
    n = 64
    truth, mask = make_phantom(n, lesion=True)
    l_roi, bg_roi = lesion_rois(n)

    res = score(truth, truth, mask, l_roi, bg_roi)
    assert set(res.keys()) == {"psnr", "ssim", "lesion_contrast", "background_noise", "contrast_recovery"}
    assert res["psnr"] == float("inf")
    assert res["ssim"] == 1.0
    np.testing.assert_allclose(res["lesion_contrast"], 0.25, atol=1e-12)
    np.testing.assert_allclose(res["contrast_recovery"], 1.0, atol=1e-12)
    np.testing.assert_allclose(res["background_noise"], 0.0, atol=1e-12)
