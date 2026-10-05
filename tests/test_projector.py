# test_projector.py: projector and adjoint tests (Step 2, strengthened in Step 2b).
"""Unit tests for discrete Radon projection operator A and CountingOperator wrapper."""

import time
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import rotate
import skimage.transform

from projector import build_operator, CountingOperator
from phantom import make_phantom, make_fine_phantom

ROOT = Path(__file__).resolve().parent.parent


def _smooth_asymmetric(n: int) -> np.ndarray:
    """Offset anisotropic Gaussian [n, n], zero outside a circle of radius n/2 - 2."""
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[:n, :n]
    mask = ((yy - c) ** 2 + (xx - c) ** 2) <= (n / 2.0 - 2.0) ** 2
    return np.exp(-((xx - c - 4.0) ** 2 / 50.0 + (yy - c + 6.0) ** 2 / 30.0)) * mask


def test_counting_operator_wrapper_and_adjoint():
    """Wrapper sanity check only: A^T is A.T by construction, so this cannot catch a wrong A."""
    n = 32
    thetas = np.linspace(0, 180, 10, endpoint=False)
    A = build_operator(n, thetas)
    op = CountingOperator(A)

    assert op.n_forward == 0
    assert op.n_back == 0
    assert op.shape == A.shape

    rng = np.random.RandomState(42)
    x = rng.randn(n * n)
    y = rng.randn(A.shape[0])

    Ax = op.forward(x)
    assert op.n_forward == 1
    ATy = op.back(y)
    assert op.n_back == 1

    inner_left = np.dot(Ax, y)
    inner_right = np.dot(x, ATy)
    rel_err = abs(inner_left - inner_right) / abs(inner_left)
    assert rel_err < 1e-10


def test_counting_operator_rejects_wrong_size():
    """forward/back raise ValueError on wrongly sized input and do not count the call."""
    n = 16
    op = CountingOperator(build_operator(n, [0.0, 90.0]))
    with pytest.raises(ValueError):
        op.forward(np.ones(n * n + 1))
    with pytest.raises(ValueError):
        op.forward(np.ones((n, n)))
    with pytest.raises(ValueError):
        op.back(np.ones(op.shape[0] - 1))
    assert op.n_forward == 0
    assert op.n_back == 0


def test_scalar_theta_accepted():
    """A scalar angle behaves like a one-element list."""
    A_scalar = build_operator(32, 45.0)
    A_list = build_operator(32, [45.0])
    assert A_scalar.shape == (34, 32 * 32)
    assert (A_scalar != A_list).nnz == 0


def test_column_sums_inside_mask():
    """(a) Every pixel whose centre is inside the mask has column sum equal to n_angles."""
    n = 64
    thetas = np.linspace(0, 180, 24, endpoint=False)
    A = build_operator(n, thetas)  # n_det = n + 2 by default
    _, mask = make_phantom(n)

    col_sums = np.array(A.sum(axis=0)).ravel()
    np.testing.assert_allclose(col_sums[mask.ravel()], len(thetas), atol=1e-12)


def test_per_angle_mass_conservation():
    """(b) For an image supported in the mask, every projection sums to the image sum."""
    n = 64
    thetas = np.linspace(0, 180, 30, endpoint=False)
    n_det = n + 2
    A = build_operator(n, thetas, n_det=n_det)
    img, _ = make_phantom(n, lesion=True)

    sino = (A @ img.ravel()).reshape(len(thetas), n_det)
    rel_diffs = np.abs(sino.sum(axis=1) - img.sum()) / img.sum()
    assert np.max(rel_diffs) < 1e-12


def test_single_pixel_hardcoded_literals():
    """(c1) Hand-derived literals: n=32, n_det=34, pixel (r=10, c=20) -> x=4.5, y=-5.5."""
    n, n_det = 32, 34
    A = build_operator(n, [0.0, 30.0, 60.0, 90.0], n_det=n_det)
    pix = 10 * n + 20
    col = A[:, pix].toarray().ravel()
    assert col[0 * 34 + 21] == pytest.approx(1.0, abs=1e-6)       # theta=0:  u = 4.5 + 16.5
    assert col[3 * 34 + 11] == pytest.approx(1.0, abs=1e-6)       # theta=90: u = -5.5 + 16.5
    assert col[1 * 34 + 17] == pytest.approx(0.352886, abs=1e-6)  # theta=30: u = 17.647114
    assert col[1 * 34 + 18] == pytest.approx(0.647114, abs=1e-6)
    for a in range(4):
        assert col[a * 34:(a + 1) * 34].sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("r, c", [(10, 20), (0, 15), (31, 16)])
def test_single_pixel_analytic_location_and_weights(r, c):
    """(c2) Single pixel lands on bins floor(u), floor(u)+1 with linear weights, many angles."""
    n, n_det = 32, 34
    c_img, c_det = (n - 1) / 2.0, (n_det - 1) / 2.0
    angles_deg = [0.0, 17.0, 30.0, 45.0, 73.0, 90.0, 135.0, 179.0]
    A = build_operator(n, angles_deg, n_det=n_det)
    col = A[:, r * n + c].toarray().ravel()

    for i, theta_deg in enumerate(angles_deg):
        t = np.deg2rad(theta_deg)
        u = (c - c_img) * np.cos(t) + (r - c_img) * np.sin(t) + c_det
        b0 = int(np.floor(u))
        expected = np.zeros(n_det)
        expected[b0] += 1.0 - (u - b0)
        if b0 + 1 < n_det:
            expected[b0 + 1] += u - b0
        np.testing.assert_allclose(col[i * n_det:(i + 1) * n_det], expected, atol=1e-12)


def test_cross_check_reference_prototype_rotate():
    """(d) Default n+2 operator, guard bins dropped, vs prototype rotate-and-sum forward()."""
    n = 64
    img = _smooth_asymmetric(n)
    thetas = np.linspace(0, 180, 30, endpoint=False)
    p_proto = np.stack([rotate(img, t, reshape=False, order=1).sum(axis=0) for t in thetas])

    A = build_operator(n, thetas)
    p_op = (A @ img.ravel()).reshape(len(thetas), n + 2)[:, 1:-1]

    rel_l2 = np.linalg.norm(p_op - p_proto) / np.linalg.norm(p_proto)
    # T3: measured 0.003556; threshold 0.010 (~2.8x margin)
    assert rel_l2 < 0.010


def test_cross_check_skimage_radon():
    """(e) Default n+2 operator, guard bins dropped, vs skimage radon on the y-flipped image (D30)."""
    n = 64
    img = _smooth_asymmetric(n)
    thetas = np.linspace(0, 180, 30, endpoint=False)

    A = build_operator(n, thetas)
    p_op = (A @ img.ravel()).reshape(len(thetas), n + 2)[:, 1:-1]

    # radon(image, theta=None, circle=True, *, preserve_range=False); y points up in radon.
    p_radon = skimage.transform.radon(img[::-1, :], theta=thetas, circle=True).T

    corr = np.corrcoef(p_op.ravel(), p_radon.ravel())[0, 1]
    # T4: measured 0.98552; threshold 0.95 (margin 0.035); without the y-flip it is 0.208
    assert corr > 0.95


def test_cross_check_skimage_radon_odd_n():
    """(Step 2c) Odd n=63 cross-check: rotation centre index n//2 coincides with (n-1)/2.0."""
    n = 63
    img = _smooth_asymmetric(n)
    thetas = np.linspace(0, 180, 30, endpoint=False)

    A = build_operator(n, thetas)
    p_op = (A @ img.ravel()).reshape(len(thetas), n + 2)[:, 1:-1]
    p_radon = skimage.transform.radon(img[::-1, :], theta=thetas, circle=True).T

    corr = np.corrcoef(p_op.ravel(), p_radon.ravel())[0, 1]
    # Measured 0.999992; threshold 0.999
    assert corr > 0.999


def test_centred_disk_symmetry_and_invariance():
    """(f) Centred disk: symmetric profiles that agree across angles within measured tolerance."""
    n = 64
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[:n, :n]
    disk = (((xx - c) ** 2 + (yy - c) ** 2) <= 15.0 ** 2).astype(float)

    thetas = np.linspace(0, 180, 30, endpoint=False)
    n_det = n + 2
    A = build_operator(n, thetas, n_det=n_det)
    p_disk = (A @ disk.ravel()).reshape(len(thetas), n_det)

    assert np.max(np.abs(p_disk - p_disk[:, ::-1])) < 1e-12  # T6

    mean_profile = p_disk.mean(axis=0)
    rel_variation = np.max(np.abs(p_disk - mean_profile)) / mean_profile.max()
    assert rel_variation < 0.05  # T5: measured 0.0305


def test_fine_operator_consistency_and_pair_binning():
    """(g) build_operator(2n, thetas, n_det=2*(n+2)) pair-summed gives n+2 bins and conserves mass."""
    n = 32
    thetas = np.linspace(0, 180, 15, endpoint=False)
    fine = make_fine_phantom(n, lesion=True)

    n_fine_det = 2 * (n + 2)
    A_fine = build_operator(2 * n, thetas, n_det=n_fine_det)
    p_fine = (A_fine @ fine.ravel()).reshape(len(thetas), n_fine_det)

    p_paired = p_fine[:, 0::2] + p_fine[:, 1::2]
    assert p_paired.shape == (len(thetas), n + 2)

    rel_diffs = np.abs(p_paired.sum(axis=1) - fine.sum()) / fine.sum()
    assert np.max(rel_diffs) < 1e-12


def test_operator_structure():
    """(h1) Shape, <= 2 nonzeros per angle per column, nnz bound, no stored zeros, sorted indices."""
    n, n_angles = 128, 60
    A = build_operator(n, np.linspace(0, 180, n_angles, endpoint=False))

    assert A.shape == (n_angles * (n + 2), n * n)
    assert A.dtype == np.float64
    col_nnz = np.diff(A.tocsc().indptr)
    assert col_nnz.max() <= 2 * n_angles
    assert A.nnz <= 2 * n * n * n_angles
    assert np.all(A.data != 0.0)
    assert A.has_sorted_indices


def test_operator_benchmark_report():
    """(h2) Report shape, nnz, build/forward/back median times (5 runs); no wall-clock asserts."""
    n, n_angles = 128, 60
    thetas = np.linspace(0, 180, n_angles, endpoint=False)

    builds = []
    for _ in range(5):
        t0 = time.perf_counter()
        A = build_operator(n, thetas)
        builds.append(time.perf_counter() - t0)

    op = CountingOperator(A)
    x = np.ones(A.shape[1])
    y = np.ones(A.shape[0])
    fwd, back = [], []
    for _ in range(5):
        t0 = time.perf_counter()
        op.forward(x)
        fwd.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        op.back(y)
        back.append(time.perf_counter() - t0)

    lines = [
        f"n={n} n_angles={n_angles} n_det={n + 2}",
        f"shape={A.shape}",
        f"nnz={A.nnz}",
        f"build_s_median5={np.median(builds):.4f}",
        f"forward_ms_median5={np.median(fwd) * 1e3:.3f}",
        f"back_ms_median5={np.median(back) * 1e3:.3f}",
    ]
    text = "\n".join(lines) + "\n"
    print(text)
    (ROOT / "results" / "benchmark_projector.txt").write_text(text, encoding="utf-8")
    assert op.n_forward == 5 and op.n_back == 5
