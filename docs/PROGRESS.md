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

```
> .venv\Scripts\python.exe -m pytest -q
............                                                             [100%]
12 passed in 2.18s
```



