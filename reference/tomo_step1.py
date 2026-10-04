#!/usr/bin/env python3
"""
Step 1: low-count tomography, FBP vs MLEM  (pure Python, runs on a laptop CPU)

What it does
  1. Takes a test image (Shepp-Logan phantom) as the "true" body slice.
  2. Simulates a scan: forward projection A x = sinogram, then Poisson noise
     (few counts = low dose / short scan).
  3. Reconstructs with
       - FBP : 1D ramp filter in Fourier space + back-projection (what you know)
       - MLEM: the iterative update from your slide
  4. Prints PSNR/SSIM and saves a comparison figure.

Install:  pip install numpy scipy scikit-image matplotlib
Run:      python tomo_step1.py
          python tomo_step1.py --angles 30 --counts 50000 --iters 150
"""
import argparse

import numpy as np
from scipy.ndimage import rotate
from skimage.data import shepp_logan_phantom
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# The two operators of the slide:  A (forward projection) and A^T (back-projection)
# ----------------------------------------------------------------------------
def forward(x, thetas):
    """(A x)_i : rotate the image, sum along one axis -> one projection per angle."""
    return np.stack([rotate(x, t, reshape=False, order=1).sum(axis=0) for t in thetas])


def back(sino, thetas, n):
    """A^T y : smear each projection back across the image, rotate, and add up."""
    out = np.zeros((n, n))
    for p, t in zip(sino, thetas):
        out += rotate(np.tile(p, (n, 1)), -t, reshape=False, order=1)
    return out


# ----------------------------------------------------------------------------
# Method 1: filtered back-projection (your 1D Fourier filtering + back-projection)
# ----------------------------------------------------------------------------
def fbp(sino, thetas, n, hann=True):
    m = sino.shape[1]
    size = 2 * m                                   # zero-pad: avoids FFT wrap-around
    freq = np.fft.fftfreq(size)                    # cycles per pixel
    filt = np.abs(freq)                            # ramp filter |omega|
    if hann:                                       # window: tames noise amplification
        filt = filt * 0.5 * (1 + np.cos(2 * np.pi * freq))
    q = np.real(np.fft.ifft(np.fft.fft(sino, n=size, axis=1) * filt, axis=1))[:, :m]
    return back(q, thetas, n) * np.pi / len(thetas)


# ----------------------------------------------------------------------------
# Method 2: MLEM (the slide)
#   ln L(x|m) = sum_i ( m_i ln(Ax)_i - (Ax)_i )          Poisson log-likelihood
#   gradient_j = [A^T (m/Ax)]_j - [A^T 1]_j              red box on the slide
#   plain gradient ascent:  x <- x + gradient            (can go negative)
#   MLEM = gradient ascent with a per-pixel step x/(A^T 1):
#       x <- x + (x / A^T 1) * (A^T(m/Ax) - A^T 1)  =  x / (A^T 1) * A^T(m/Ax)
# ----------------------------------------------------------------------------
def mlem(m, thetas, n, mask, n_iter, truth_scaled=None, scale=1.0):
    sens = np.maximum(back(np.ones_like(m), thetas, n), 1e-6)    # A^T 1
    x = mask.astype(float)                                       # flat start
    history = []
    for _ in range(n_iter):
        ratio = m / np.maximum(forward(x, thetas), 1e-9)         # m_i / (Ax)_i
        x = x / sens * back(ratio, thetas, n) * mask             # multiplicative update
        if truth_scaled is not None:
            history.append(psnr(truth_scaled, np.clip(x / scale, 0, 1), data_range=1))
    return x, history


def circle_mask(n):
    yy, xx = np.mgrid[:n, :n]
    c = (n - 1) / 2
    return ((yy - c) ** 2 + (xx - c) ** 2 <= (n / 2) ** 2).astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=128, help="image size (pixels)")
    ap.add_argument("--angles", type=int, default=60, help="number of projection angles")
    ap.add_argument("--counts", type=float, default=2e5, help="total counts (lower = lower dose)")
    ap.add_argument("--iters", type=int, default=100, help="MLEM iterations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="step1_result.png")
    args = ap.parse_args()

    n = args.size
    thetas = np.linspace(0, 180, args.angles, endpoint=False)
    mask = circle_mask(n)
    truth = resize(shepp_logan_phantom(), (n, n), anti_aliasing=True) * mask

    # --- simulate the scan -------------------------------------------------
    sino_clean = forward(truth, thetas)
    scale = args.counts / sino_clean.sum()                  # sets the dose
    rng = np.random.default_rng(args.seed)
    m = rng.poisson(sino_clean * scale).astype(float)       # noisy measured counts m_i

    # --- reconstruct ---------------------------------------------------------
    rec_fbp = np.clip(fbp(m, thetas, n) / scale * mask, 0, 1)
    rec_mlem, hist = mlem(m, thetas, n, mask, args.iters, truth, scale)
    rec_mlem = np.clip(rec_mlem / scale, 0, 1)
    best_k = int(np.argmax(hist)) + 1

    def report(name, rec):
        print(f"{name:<22} PSNR {psnr(truth, rec, data_range=1):6.2f} dB   "
              f"SSIM {ssim(truth, rec, data_range=1):.3f}")

    print(f"\n{args.angles} angles, {args.counts:,.0f} total counts, {n}x{n} image")
    report("FBP (ramp + Hann)", rec_fbp)
    report(f"MLEM, {args.iters} iterations", rec_mlem)
    print(f"MLEM PSNR peaks at iteration {best_k} ({hist[best_k - 1]:.2f} dB), "
          f"then {'falls (noise grows)' if best_k < args.iters else 'is still rising'}")

    # --- figure ----------------------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    ax[0, 0].imshow(truth, cmap="gray"); ax[0, 0].set_title("True image")
    ax[0, 1].imshow(m, cmap="gray", aspect="auto"); ax[0, 1].set_title("Noisy sinogram m_i")
    ax[0, 2].imshow(rec_fbp, cmap="gray"); ax[0, 2].set_title("FBP")
    ax[1, 0].imshow(rec_mlem, cmap="gray"); ax[1, 0].set_title(f"MLEM ({args.iters} it.)")
    ax[1, 1].imshow(np.abs(rec_mlem - truth), cmap="magma"); ax[1, 1].set_title("MLEM error map")
    ax[1, 2].plot(range(1, len(hist) + 1), hist, label="MLEM")
    ax[1, 2].axhline(psnr(truth, rec_fbp, data_range=1), ls="--", c="r", label="FBP")
    ax[1, 2].set_xlabel("iteration"); ax[1, 2].set_ylabel("PSNR (dB)")
    ax[1, 2].legend(); ax[1, 2].set_title("Quality vs iterations")
    for a in ax.ravel()[:5]:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"Saved figure: {args.out}")


if __name__ == "__main__":
    main()
