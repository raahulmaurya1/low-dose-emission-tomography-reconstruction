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
