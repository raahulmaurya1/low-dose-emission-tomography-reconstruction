# test_methods.py: FBP, MLEM, OSEM, and MLEM+TV tests (Steps 4-7).
# Takes: methods.py with projector.py and scan.py.
# Returns: pass/fail via pytest.

import numpy as np
import pytest

from config import Settings
from phantom import make_fine_phantom, make_phantom, block_mean, lesion_rois
from projector import build_operator, CountingOperator
from scan import simulate_scan
from metrics import psnr


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def setup_64():
    """Shared setup at n=64 for FBP tests."""
    n = 64
    truth, mask = make_phantom(n, lesion=True)
    fine_truth = make_fine_phantom(n, lesion=True)
    _, bg_roi = lesion_rois(n)
    return n, truth, mask, fine_truth, bg_roi


@pytest.fixture(scope="module")
def noisefree_180(setup_64):
    """Noise-free expected sinogram at 180 angles for amplitude/PSNR tests."""
    n, truth, mask, fine_truth, bg_roi = setup_64
    n_angles = 180
    thetas = np.linspace(0, 180, n_angles, endpoint=False)
    _, scale, expected = simulate_scan(fine_truth, thetas, counts=1e9, seed=0)
    m_noisefree = scale * expected
    op = CountingOperator(build_operator(n, thetas))
    return op, m_noisefree, thetas, scale, n_angles


# ---------------------------------------------------------------------------
# Test (a): FBP amplitude – mean inside background ROI vs truth 0.20
# ---------------------------------------------------------------------------

def test_fbp_amplitude_background_roi(setup_64, noisefree_180):
    """Noise-free FBP at 180 angles: mean inside background ROI matches truth within tolerance.

    T11: Measured ratio = 0.980 (-2.0%), consistent with D34 (4*n_det padding).
    Threshold: |ratio - 1.0| < 0.04 (margin ~2x over measured 0.020 deviation).
    """
    from methods import fbp

    n, truth, mask, fine_truth, bg_roi = setup_64
    op, m_noisefree, thetas, scale, n_angles = noisefree_180

    rec_data = fbp(op, m_noisefree, thetas, mask, window="hann")
    rec_scaled = np.clip(rec_data / scale, 0, 1)

    truth_mean = float(np.mean(truth[bg_roi]))
    rec_mean = float(np.mean(rec_scaled[bg_roi]))
    ratio = rec_mean / truth_mean

    print(f"\n[amplitude] truth_mean={truth_mean:.6f}  rec_mean={rec_mean:.6f}  ratio={ratio:.6f}")

    # T11: measured 0.980, threshold 0.04 (margin 2x over 0.020 deviation)
    assert abs(ratio - 1.0) < 0.04, (
        f"FBP amplitude ratio {ratio:.4f} deviates from 1.0 by more than 4%"
    )


# ---------------------------------------------------------------------------
# Test (b): Noise-free PSNR at 180 angles with floor; print PSNR at multiple angles
# ---------------------------------------------------------------------------

def test_fbp_noisefree_psnr(setup_64, noisefree_180):
    """Noise-free FBP PSNR at 180 angles exceeds a measured floor.

    T12: Measured noise-free PSNR at 180 angles = 21.03 dB.
    Floor: 19.0 dB (margin ~2 dB below measured).
    PSNR at 15/30/60/90/180 angles: 20.55/21.02/21.03/21.03/21.03 dB.
    Monotonicity NOT asserted: PSNR is flat across 30-180 angles (discretisation-limited).
    """
    from methods import fbp

    n, truth, mask, fine_truth, bg_roi = setup_64
    op, m_noisefree, thetas, scale, n_angles = noisefree_180

    rec_data = fbp(op, m_noisefree, thetas, mask, window="hann")
    rec_scaled = np.clip(rec_data / scale, 0, 1)
    psnr_180 = psnr(truth, rec_scaled, mask)

    print(f"\n[psnr] 180 angles: {psnr_180:.2f} dB")

    # Print PSNR for 15, 30, 60, 90 angles
    for na in [15, 30, 60, 90]:
        th = np.linspace(0, 180, na, endpoint=False)
        _, sc, exp = simulate_scan(fine_truth, th, counts=1e9, seed=0)
        m_nf = sc * exp
        op_temp = CountingOperator(build_operator(n, th))
        rec_temp = fbp(op_temp, m_nf, th, mask, window="hann")
        rec_temp_sc = np.clip(rec_temp / sc, 0, 1)
        p = psnr(truth, rec_temp_sc, mask)
        print(f"[psnr] {na:3d} angles: {p:.2f} dB")

    # T12: measured 21.03 dB; floor 19.0 dB (margin ~2 dB)
    assert psnr_180 > 19.0, f"Noise-free FBP PSNR {psnr_180:.2f} dB is below floor 19.0"


# ---------------------------------------------------------------------------
# Test (c): Linearity – fbp(a*m1 + b*m2) = a*fbp(m1) + b*fbp(m2)
# ---------------------------------------------------------------------------

def test_fbp_linearity(setup_64):
    """FBP is a linear operation. Max abs diff measured at 8.88e-15."""
    from methods import fbp

    n, truth, mask, fine_truth, bg_roi = setup_64
    n_angles = 60
    thetas = np.linspace(0, 180, n_angles, endpoint=False)
    op = CountingOperator(build_operator(n, thetas))

    rng = np.random.default_rng(42)
    n_det = n + 2
    m1 = rng.random((n_angles, n_det)).astype(np.float64) * 100
    m2 = rng.random((n_angles, n_det)).astype(np.float64) * 100
    a, b = 2.3, 0.7

    rec1 = fbp(op, m1, thetas, mask, window="hann")
    rec2 = fbp(op, m2, thetas, mask, window="hann")
    rec_combined = fbp(op, a * m1 + b * m2, thetas, mask, window="hann")

    diff = np.max(np.abs(rec_combined - (a * rec1 + b * rec2)))
    print(f"\n[linearity] max abs diff = {diff:.2e}")
    assert diff < 1e-10, f"FBP linearity violated: max diff = {diff:.2e}"


# ---------------------------------------------------------------------------
# Test (d): Impulse response – argmax of FBP of single-pixel projections
# ---------------------------------------------------------------------------

def test_fbp_impulse_argmax(setup_64):
    """FBP of single-pixel projections peaks at the correct pixel."""
    from methods import fbp

    n, truth, mask, fine_truth, bg_roi = setup_64
    n_angles = 60
    thetas = np.linspace(0, 180, n_angles, endpoint=False)
    A = build_operator(n, thetas)
    op = CountingOperator(A)
    n_det = n + 2

    c = (n - 1) / 2.0
    test_pixels = [
        (n // 2, n // 2),           # centre
        (n // 4, n // 4),           # upper-left quadrant
        (3 * n // 4, n // 4),       # lower-left quadrant
        (n // 4, 3 * n // 4),       # upper-right quadrant
        (n // 2, n // 4),           # left of centre
    ]

    for r, col in test_pixels:
        if (r - c) ** 2 + (col - c) ** 2 > (n / 2.0) ** 2:
            continue
        img = np.zeros((n, n), dtype=np.float64)
        img[r, col] = 1.0
        m_pix = (A @ img.ravel()).reshape(n_angles, n_det)
        rec = fbp(op, m_pix, thetas, mask, window="hann")
        peak = np.unravel_index(np.argmax(rec), (n, n))
        print(f"\n[impulse] pixel ({r},{col}) -> peak at {peak}")
        assert peak == (r, col), (
            f"FBP impulse peak at {peak}, expected ({r},{col})"
        )


# ---------------------------------------------------------------------------
# Test (e): Window effect – std inside background ROI: "none" > "hann"
# ---------------------------------------------------------------------------

def test_fbp_window_noise_reduction(setup_64):
    """Hann window reduces noise vs no window on Poisson data.

    T13: Measured std ratio (none/hann) = 2.520.
    """
    from methods import fbp

    n, truth, mask, fine_truth, bg_roi = setup_64
    n_angles = 60
    thetas = np.linspace(0, 180, n_angles, endpoint=False)
    op = CountingOperator(build_operator(n, thetas))

    m, scale, expected = simulate_scan(fine_truth, thetas, counts=2e5, seed=0)

    rec_hann = fbp(op, m, thetas, mask, window="hann")
    rec_none = fbp(op, m, thetas, mask, window="none")

    rec_hann_sc = np.clip(rec_hann / scale, 0, 1)
    rec_none_sc = np.clip(rec_none / scale, 0, 1)

    std_hann = float(np.std(rec_hann_sc[bg_roi]))
    std_none = float(np.std(rec_none_sc[bg_roi]))
    ratio = std_none / std_hann

    print(f"\n[window] std_hann={std_hann:.6f}  std_none={std_none:.6f}  ratio={ratio:.3f}")

    assert std_none > std_hann, (
        f"Expected std_none ({std_none:.6f}) > std_hann ({std_hann:.6f})"
    )
