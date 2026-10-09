# PROGRESS.md

One evidence block per step. A block is marked APPROVED only after the owner approves it.

---

## Step 0: Setup (status: APPROVED by owner, 2026-10-05)

### Environment

```
> .venv\Scripts\python.exe --version
Python 3.12.10

> git --version
git version 2.54.0.windows.1

> .venv\Scripts\python.exe -m pip list --format=freeze   (filtered to the 5 direct packages)
matplotlib==3.11.2
numpy==2.5.3
pytest==9.1.1
scikit-image==0.26.0
scipy==1.18.1
```

### git init

```
> git init
Initialized empty Git repository in D:/sparse-view-low-dose-tomographic-reconstruction-final/.git/
```

### Reference prototype (reference/tomo_step1.py, run unchanged)

Figures were written outside the repository (D4).

```
> .venv\Scripts\python.exe reference\tomo_step1.py --out <scratch>\proto_60_2e5.png

60 angles, 200,000 total counts, 128x128 image
FBP (ramp + Hann)      PSNR  20.36 dB   SSIM 0.403
MLEM, 100 iterations   PSNR  17.20 dB   SSIM 0.574
MLEM PSNR peaks at iteration 20 (22.72 dB), then falls (noise grows)
elapsed_s: 51.7830385

> .venv\Scripts\python.exe reference\tomo_step1.py --angles 30 --counts 3e4 --out <scratch>\proto_30_3e4.png

30 angles, 30,000 total counts, 128x128 image
FBP (ramp + Hann)      PSNR  14.63 dB   SSIM 0.264
MLEM, 100 iterations   PSNR  13.50 dB   SSIM 0.493
MLEM PSNR peaks at iteration 10 (19.22 dB), then falls (noise grows)
elapsed_s: 13.8131382
```

Note: the prototype computes PSNR/SSIM over the whole image with skimage defaults, not with the
D9/D10 mask-based definitions, so its numbers are not directly comparable to this project's.

### Gate: pytest runs

```
> .venv\Scripts\python.exe -m pytest -q

no tests ran in 0.07s
pytest exit code: 5

> .venv\Scripts\python.exe -m pytest --collect-only
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\sparse-view-low-dose-tomographic-reconstruction-final
configfile: pytest.ini
testpaths: tests
collected 0 items
```

Exit code 5 is pytest's "no tests collected" code; zero tests is allowed by the Step 0 gate.

---

## Step 1: config, phantom, metrics (status: APPROVED by owner, 2026-10-05)

### Library signatures verified

```python
# skimage.transform.resize:
resize(image, output_shape, order=None, mode='reflect', cval=0, clip=True, preserve_range=False, anti_aliasing=None, anti_aliasing_sigma=None)

# skimage.metrics.structural_similarity:
structural_similarity(im1, im2, *, win_size=None, gradient=False, data_range=None, channel_axis=None, gaussian_weights=False, full=False, **kwargs)

# skimage.data.shepp_logan_phantom:
shepp_logan_phantom()

# scipy.ndimage.binary_erosion:
binary_erosion(input, structure=None, iterations=1, mask=None, output=None, border_value=0, origin=0, brute_force=False, *, axes=None)
```

### Gate: failing test run (before implementation)

```
> .venv\Scripts\python.exe -m pytest -q
=================================== ERRORS ====================================
____________________ ERROR collecting tests/test_basics.py ____________________
ImportError while importing test module 'D:\sparse-view-low-dose-tomographic-reconstruction-final\tests\test_basics.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Users\Dell\AppData\Local\Programs\Python\Python312\Lib\importlib\__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\test_basics.py:5: in <module>
    from config import Settings
E   ImportError: cannot import name 'Settings' from 'config' (D:\sparse-view-low-dose-tomographic-reconstruction-final\src\config.py)
=========================== short test summary info ===========================
ERROR tests/test_basics.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 1.60s
```

### Gate: passing test run (after initial implementation)

```
> .venv\Scripts\python.exe -m pytest -q
...........                                                              [100%]
11 passed in 1.80s
```

### Revision round (2026-10-05)

- Lesion contrast parameter added (D26), defaulting to +25% (+0.05 elevation over 0.20 tissue, intensity 0.25).
- Background ROI enlarged to radius 10 at n=128 (radius 5 at n=64) with 1 px erosion (D27), yielding 248 pixels at n=128 (>= 150) and 44 pixels at n=64.
- Mutation checks:
  1. Mutated psnr to average over whole image -> test_psnr_metrics failed (ACTUAL: 21.03 dB vs DESIRED: 20.0 dB).
  2. Mutated lesion_contrast to swap lesion and background -> 5 tests failed.
  Code restored; `git diff src/` confirmed clean.
- ROI outline verification figure saved to `results/phantom_rois.png`.

### Revision round 2 (2026-10-05)

- Spaced ROIs: background ROI moved to lower-left uniform brain tissue at (33*n/128, -20*n/128), radius 8.5*n/128.
  - Pixel count at n=128: 164 px (>= 150).
  - Measured gap between ROI boundaries: 39.56 px at n=128 (>= 8 px), 21.19 px at n=64 (>= 4 px).
  - Measured distance to any intensity edge: background ROI 5.00 px (>= 5 px), lesion ROI 7.21 px (>= 5 px).
  - Standard deviation in clean truth: <= 5.6e-17.
- Mutation proof with SHA256 hashes:
  - Baseline SHA256 of `src\metrics.py`: `A6E46EAF37216FC5941C42E1A118BCA966650788B6AAAF49643EC86ACAB80745`
  - Mutated psnr to whole-image average -> `test_psnr_metrics` failed with `ACTUAL: 21.034264` vs `DESIRED: 20.0`.
  - Restored `src\metrics.py` -> SHA256: `A6E46EAF37216FC5941C42E1A118BCA966650788B6AAAF49643EC86ACAB80745` (identical).
- Regenerated figure `results/phantom_rois.png` reflecting spaced ROIs.

### Final test run (after revision 2)

---

## Step 2: projector (status: APPROVED by owner, 2026-10-05)

### Library signatures verified

```python
# skimage.transform.radon:
radon(image, theta=None, circle=True, *, preserve_range=False)

# scipy.sparse.csr_matrix:
csr_matrix((data, (row_ind, col_ind)), shape=(M, N), dtype=None)
```

### Gate: failing test run (before implementation)

```
> .venv\Scripts\python.exe -m pytest -q tests/test_projector.py

=================================== ERRORS ====================================
__________________ ERROR collecting tests/test_projector.py ___________________
ImportError while importing test module 'D:\sparse-view-low-dose-tomographic-reconstruction-final\tests\test_projector.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Users\Dell\AppData\Local\Programs\Python\Python312\Lib\importlib\__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\test_projector.py:10: in <module>
    from projector import build_operator, CountingOperator
E   ImportError: cannot import name 'build_operator' from 'projector' (D:\sparse-view-low-dose-tomographic-reconstruction-final\src\projector.py)
=========================== short test summary info ===========================
ERROR tests/test_projector.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 1.92s
```

### Gate: passing test run (after implementation)

```
> .venv\Scripts\python.exe -m pytest -v tests/test_projector.py

============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.6.0 -- D:\sparse-view-low-dose-tomographic-reconstruction-final\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\sparse-view-low-dose-tomographic-reconstruction-final
configfile: pytest.ini
collecting ... collected 9 items

tests/test_projector.py::test_counting_operator_wrapper_and_adjoint PASSED [ 11%]
tests/test_projector.py::test_column_sums_inside_mask PASSED             [ 22%]
tests/test_projector.py::test_per_angle_mass_conservation PASSED         [ 33%]
tests/test_projector.py::test_single_pixel_analytic_location_and_weights PASSED [ 44%]
tests/test_projector.py::test_cross_check_reference_prototype_rotate PASSED [ 55%]
tests/test_projector.py::test_cross_check_skimage_radon PASSED           [ 66%]
tests/test_projector.py::test_centred_disk_symmetry_and_invariance PASSED [ 77%]
tests/test_projector.py::test_fine_operator_consistency_and_pair_binning PASSED [ 88%]
tests/test_projector.py::test_operator_shape_sparsity_and_benchmark PASSED [100%]

============================== 9 passed in 3.04s ==============================
```

### Mutation checks with SHA-256

- Baseline SHA-256 of `src\projector.py`:
  `B1392499FF3132354F70FBCD55CB3D35E54ADE7DF3CAC754EA83A3B2E32814D1`

1. **Mutation (1): Shift s by +0.5**
   - Modified `s = ... + 0.5`
   - pytest output:
     `FAILED tests/test_projector.py::test_single_pixel_analytic_location_and_weights`
     `FAILED tests/test_projector.py::test_cross_check_reference_prototype_rotate`
     `FAILED tests/test_projector.py::test_centred_disk_symmetry_and_invariance`
     `3 failed, 6 passed in 3.48s`

2. **Mutation (2): Negate the angle**
   - Modified `thetas_rad = -np.deg2rad(thetas_deg_arr)`
   - pytest output:
     `FAILED tests/test_projector.py::test_single_pixel_analytic_location_and_weights`
     `FAILED tests/test_projector.py::test_cross_check_reference_prototype_rotate`
     `2 failed, 7 passed in 2.81s`

3. **Mutation (3): Nearest-bin only (drop linear weight)**
   - Modified `b_near = np.round(u).astype(np.int64); vals = 1.0`
   - pytest output:
     `FAILED tests/test_projector.py::test_single_pixel_analytic_location_and_weights`
     `FAILED tests/test_projector.py::test_cross_check_reference_prototype_rotate`
     `FAILED tests/test_projector.py::test_centred_disk_symmetry_and_invariance`
     `3 failed, 6 passed in 2.82s`

- Restored file and re-computed SHA-256 of `src\projector.py`:
  `B1392499FF3132354F70FBCD55CB3D35E54ADE7DF3CAC754EA83A3B2E32814D1` (IDENTICAL)

### Full test suite run (after restoration)

```
> .venv\Scripts\python.exe -m pytest -q
.....................                                                    [100%]
21 passed in 3.27s
```

### Operator measurements (n=128, 60 angles, n_det=130)

- **Shape**: (7800, 16384)
- **Non-zero entries (nnz)**: 1,863,160
- **Density**: 1.458%
- **Sparsity**: 98.542%
- **Build time**: 368.90 ms
- **Forward product time (A @ x)**: 5.043 ms
- **Back product time (A.T @ y)**: 4.764 ms

---

## Step 2b: projector test strengthening + small fixes (status: APPROVED by owner, 2026-10-05)

### Failing run (new tests, Step 2 code)

```
> .venv\Scripts\python.exe -m pytest -q tests/test_projector.py
FAILED tests/test_projector.py::test_counting_operator_rejects_wrong_size - assert 2 == 0 (n_forward)
FAILED tests/test_projector.py::test_scalar_theta_accepted - TypeError: object of type 'numpy.float64' has no len()
FAILED tests/test_projector.py::test_operator_structure - assert np.all(A.data != 0.0) -> False
3 failed, 12 passed in 7.12s
```

The hand-derived literal test passed on the Step 2 code (consistent with the hand arithmetic).

### Passing run (after fixes)

```
> .venv\Scripts\python.exe -m pytest -q tests/test_projector.py
15 passed in 4.13s
```

### Identity vs Step 2 reference (exact indptr/indices/data equality)

```
n= 32 angles= 1 ref_nnz_raw=    2048 ref_nnz_elim=    1024 new_nnz=    1024 identical=True
n= 32 angles= 7 ref_nnz_raw=   13804 ref_nnz_elim=   12780 new_nnz=   12780 identical=True
n= 32 angles=60 ref_nnz_raw=  118328 ref_nnz_elim=  116297 new_nnz=  116297 identical=True
n= 64 angles= 1 ref_nnz_raw=    8192 ref_nnz_elim=    4096 new_nnz=    4096 identical=True
n= 64 angles= 7 ref_nnz_raw=   54668 ref_nnz_elim=   50572 new_nnz=   50572 identical=True
n= 64 angles=60 ref_nnz_raw=  468580 ref_nnz_elim=  460488 new_nnz=  460488 identical=True
n=128 angles= 1 ref_nnz_raw=   32768 ref_nnz_elim=   16384 new_nnz=   16384 identical=True
n=128 angles= 7 ref_nnz_raw=  217488 ref_nnz_elim=  201104 new_nnz=  201104 identical=True
n=128 angles=60 ref_nnz_raw= 1863160 ref_nnz_elim= 1830797 new_nnz= 1830797 identical=True
ALL IDENTICAL: True
n=256, 180 angles, before (ref): peak=1500.3 MiB build=4.07s nnz=22285420
n=256, 180 angles, after (new): peak=512.6 MiB build=2.13s nnz=22155987
```

### Mutations (baseline SHA-256 D466C48D75CCBC947D8D77CD25E56D313607D1833CE96BF302919B69FDEB8B5D)

- (1) s + 0.5: 6 failed (literals, analytic x3, prototype 0.0807, disk symmetry 7.70). Skimage passed.
- (2) negated angle: 6 failed (literals, analytic x3, prototype 1.049, skimage r=0.340).
- (3) x/y swap: 6 failed (literals, analytic x3, prototype 1.089, skimage r=0.261).
- SHA-256 after each restore: D466C48D75CCBC947D8D77CD25E56D313607D1833CE96BF302919B69FDEB8B5D (identical).

### Benchmark (results/benchmark_projector.txt)

```
n=128 n_angles=60 n_det=130
shape=(7800, 16384)
nnz=1830797
build_s_median5=0.2870
forward_ms_median5=4.974
back_ms_median5=4.664
```

### Final full run

```
> .venv\Scripts\python.exe -m pytest -q
27 passed in 5.36s
```

---

## Step 3: scan simulation (status: APPROVED by owner, 2026-10-05)

### Library signatures verified

```python
# numpy.random.default_rng:
default_rng(seed=None)

# numpy.random.Generator.poisson:
poisson(lam=1.0, size=None)
```

### Gate: failing test run (before implementation)

```
> .venv\Scripts\python.exe -m pytest -q tests/test_basics.py

=================================== ERRORS ====================================
____________________ ERROR collecting tests/test_basics.py ____________________
ImportError while importing test module 'D:\low-dose-CT-scan-reconstructio\tests\test_basics.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Users\Dell\AppData\Local\Programs\Python\Python312\Lib\importlib\__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\test_basics.py:8: in <module>
    from scan import simulate_scan
E   ImportError: cannot import name 'simulate_scan' from 'scan' (D:\low-dose-CT-scan-reconstructio\src\scan.py)
=========================== short test summary info ===========================
ERROR tests/test_basics.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 2.75s
```

### Gate: passing test run (after implementation)

```
> .venv\Scripts\python.exe -m pytest -v tests/test_basics.py

============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.6.0 -- D:\low-dose-CT-scan-reconstructio\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\low-dose-CT-scan-reconstructio
configfile: pytest.ini
collecting ... collected 19 items

tests/test_basics.py::test_import_config_defaults PASSED                 [  5%]
tests/test_basics.py::test_phantom_shape_range_mask[64] PASSED           [ 10%]
tests/test_basics.py::test_phantom_shape_range_mask[128] PASSED          [ 15%]
tests/test_basics.py::test_fine_phantom_and_block_mean[64] PASSED        [ 21%]
tests/test_basics.py::test_fine_phantom_and_block_mean[128] PASSED       [ 26%]
tests/test_basics.py::test_lesion_and_background_rois[64] PASSED         [ 31%]
tests/test_basics.py::test_lesion_and_background_rois[128] PASSED        [ 36%]
tests/test_basics.py::test_lesion_contrast_parameter PASSED              [ 42%]
tests/test_basics.py::test_psnr_metrics PASSED                           [ 47%]
tests/test_basics.py::test_ssim_metric PASSED                            [ 52%]
tests/test_basics.py::test_contrast_and_noise_hand_computed PASSED       [ 57%]
tests/test_basics.py::test_score_dict PASSED                             [ 63%]
tests/test_basics.py::test_simulate_scan_reproducibility PASSED          [ 68%]
tests/test_basics.py::test_simulate_scan_shape_and_values PASSED         [ 73%]
tests/test_basics.py::test_simulate_scan_total_counts PASSED             [ 78%]
tests/test_basics.py::test_simulate_scan_poisson_statistics PASSED       [ 84%]
tests/test_basics.py::test_simulate_scan_mass_conservation PASSED        [ 89%]
tests/test_basics.py::test_simulate_scan_inverse_crime_guard PASSED      [ 94%]
tests/test_basics.py::test_simulate_scan_a_fine_caching_and_validation PASSED [100%]

============================= 19 passed in 5.58s ==============================
```

### Mutation checks with SHA-256

- Baseline SHA-256 of `src\scan.py`:
  `2C99DBC7B940D1264DD1700F6D6A5ED2CE99A3A408F85030219381D2A756F760`

1. **Mutation (1): Drop the `/4`**
   - Modified `fine_activity = fine_truth`
   - pytest output:
     `FAILED tests/test_basics.py::test_simulate_scan_mass_conservation - assert np.float64(45415.409824227405) < 1e-08`
     `FAILED tests/test_basics.py::test_simulate_scan_inverse_crime_guard - assert np.float64(3.0087636134555478) < 0.05`
     `2 failed, 17 passed in 3.49s`

2. **Mutation (2): Shift pair binning by one detector bin**
   - Modified `fine_proj_2d = np.roll(fine_proj.reshape(n_angles, 2 * (n + 2)), 1, axis=1)`
   - pytest output:
     `FAILED tests/test_basics.py::test_simulate_scan_inverse_crime_guard - assert np.float64(0.09641741100411627) < 0.05`
     `1 failed, 18 passed in 3.87s`

3. **Mutation (3): Replace fine pipeline by coarse operator applied to block_mean(fine_truth)**
   - Modified `expected = (build_operator(n, thetas_deg_arr, n_det=n + 2) @ block_mean(fine_truth).ravel()).reshape(n_angles, n + 2)`
   - pytest output:
     `FAILED tests/test_basics.py::test_simulate_scan_inverse_crime_guard - assert 0.005 < np.float64(0.0)`
     `1 failed, 18 passed in 2.59s`

- Restored file and re-computed SHA-256 of `src\scan.py`:
  `2C99DBC7B940D1264DD1700F6D6A5ED2CE99A3A408F85030219381D2A756F760` (IDENTICAL)

### Full test suite run (after restoration)

```
> .venv\Scripts\python.exe -m pytest -q
..................................                                       [100%]
34 passed in 5.52s
```
