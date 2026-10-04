# SPEC.md: Low-dose tomography reconstruction (read fully before doing anything)

You are a careful senior scientific-Python engineer. You build ONE small project, ONE step at a time, and you prove every step with executed evidence. You never guess. If you are unsure, you say so and ask.

---

## 1. Goal and outcome

Build a small, clean Python project that answers one question:

> How few projections / how few counts can we use and still reconstruct a trustworthy image, and which method (FBP, MLEM, OSEM, MLEM+TV) gives the best quality for the least compute?

Deliverables: tested code, a results table (mean ± std over seeds), figures drawn only from saved results, a README, and a 2-page technical note. The owner will show this project in a master's application, so correctness and honesty matter more than features.

## 2. Hard constraints

- Python only. NO C++, no Cython, no compiled extensions.
- Runs on a normal laptop CPU or free Colab/Kaggle. No GPU required. No paid services, no API keys.
- Allowed libraries ONLY: numpy, scipy, scikit-image (phantom, `skimage.transform.resize` for the phantom, and SSIM only; D7), matplotlib, pytest. No pandas; CSV uses the standard-library `csv` module (D3). Do not add anything else without asking.
- No network access at runtime. Never download datasets or models without asking me first.
- Keep it simple: the structure in section 5 (D1), plain functions, type hints, docstrings that state array shapes, functions under ~40 lines, no dead code, no global state, no frameworks.
- Deterministic: every random draw uses `numpy.random.default_rng(seed)`. Same settings + seed must give identical output.

## 3. Anti-hallucination harness (non-negotiable rules)

1. **Verify, don't recall.** Before using any library function that is not already in `reference/tomo_step1.py`, run `python -c "import x; help(x.f)"` (or read the installed source) and paste the real signature in your step report. Never assume an argument exists.
2. **No claim without execution.** Never write "tests pass", "works", or any number unless you ran the command in this session and pasted its real output.
3. **No invented numbers.** Every number in the README, note or figures must come from a file in `results/` produced by code. Never type a metric by hand. Never fabricate or "estimate" results.
4. **Tests are the oracle.** Write the test first, watch it fail, then implement. The single verification command is `python -m pytest -q`. Never weaken, skip or delete a test to make it pass. If you think a test is wrong, stop and explain why.
5. **Thresholds are measured, not guessed.** When a test needs a numeric tolerance, measure the real value first, set the threshold with a stated margin, and record both in `docs/DECISIONS.md`.
6. **Frozen interfaces.** The function signatures in section 5 are contracts. Do not change them silently. A change needs an entry in `docs/DECISIONS.md` and my approval.
7. **Scope guard.** Do only the current step. No extra files, features, refactors or dependencies. If you see something worth doing, list it under "Proposals" in the step report and do not do it.
8. **No silent fallbacks.** Never swap a method, library or dataset for an easier one. Never replace real computation with mocks or hard-coded results. If something is blocked, stop and report.
9. **Two attempts, then stop.** If a failure persists after two honest fixes, stop and report the evidence. Do not keep patching.
10. **Say "UNVERIFIED".** Anything you could not run or check is labelled UNVERIFIED in the report. Unknown is an acceptable answer; a confident guess is not.
11. **Version control.** `git init` in Step 0. Commit only after I approve a step, with message `step N: <summary>`.
12. **Stop after every step.** Output the step report (section 8) and WAIT for my approval. Never start the next step on your own.

## 4. Conventions (exact; define once, reuse everywhere)

- Image: `n x n` float64, `n=128` default (use `n=64` while developing). Pixel size 1, origin at the image center. Truth intensities in [0, 1].
- Mask: boolean inscribed circle, radius `n/2`. Reconstructions are only valid inside the mask; outside is zero.
- Angles: degrees, `numpy.linspace(0, 180, n_angles, endpoint=False)`.
- Detector: `n_det = n + 2` bins (one guard bin each side; D6), spacing 1, centered. Sinogram shape `(n_angles, n_det)`. Flattened order: angle-major.
- Operator A: `scipy.sparse.csr_matrix` of shape `(n_angles*n_det, n*n)`, **pixel-driven with linear interpolation**: each pixel center projects to detector coordinate `s = x cos(theta) + y sin(theta)` and its value is split between the two nearest bins by linear weights. `Aᵀ` is the exact transpose (`A.T`), so A and Aᵀ are a matched pair by construction.
- **No inverse crime:** the scan is simulated on a 2x finer grid (`2n x 2n` image, `2*(n+2)` detector bins via `build_operator(2n, thetas, n_det=2*(n+2))`, fine pixel value = coarse-equivalent intensity / 4), then detector bins are summed in pairs down to `n_det = n + 2` (D5, D6). Reconstruction uses the coarse A. The simulator and the reconstructor must never share an operator.
- Scaling: `expected = binned(A_fine @ fine_activity)`; `scale = counts / expected.sum()`; `m = rng.poisson(scale * expected)`. A reconstruction `x_hat` is compared to the truth as `clip(x_hat / scale, 0, 1)`. The same rescale and clip are applied to every method.
- Truth for scoring = coarse truth (2x2 block mean of the fine truth), values in [0, 1].
- PSNR uses `data_range=1.0`, hand-written, MSE over mask pixels only; identical images give inf (D9). SSIM: `skimage.metrics.structural_similarity(full=True, win_size=7, data_range=1.0, gaussian_weights=False)`, reported as the mean of the SSIM map over mask pixels (D10).
- Lesion metrics: `lesion_contrast = (mu_L - mu_B) / mu_B`; `background_noise = sigma_B / mu_B`; `contrast_recovery = contrast_rec / contrast_truth` (D11). Lesion ROI = lesion disk eroded by 1 px; background ROI = same-size disk in another uniform region, not overlapping the lesion (D12).
- Divisions `m / (A x)` use a tiny floor (`eps=1e-9`). Sensitivity image `s = Aᵀ 1` is floored at `1e-6` and masked.
- Model: Poisson counts (emission-style, as in PET/SPECT). This is NOT transmission CT physics. Parallel-beam geometry only.

## 5. File structure and interfaces

```
sparse-view-low-dose-tomographic-reconstruction/
│
├── docs/
│   ├── SPEC.md                  authoritative specification; re-read at the start of every step
│   ├── PROGRESS.md              one evidence block per approved step (maintained throughout the build)
│   └── DECISIONS.md             every deviation, measured threshold, and open question
│
├── src/                         modules import each other directly (`from config import Settings`),
│   │                            no `src.` prefix; CLI: `python src/main.py <command>`; all paths are
│   │                            built from the project root, `Path(__file__).resolve().parent.parent` (D1)
│   ├── main.py                  CLI entry point only; no logic
│   ├── config.py                Settings dataclass
│   ├── phantom.py               test images and ROIs
│   ├── projector.py             operator A / Aᵀ and the call-counting wrapper
│   ├── scan.py                  scan simulation: fine-grid projection + Poisson noise
│   ├── methods.py               fbp, mlem, osem, mlem_tv
│   ├── metrics.py               PSNR, SSIM, lesion contrast, background noise
│   ├── experiments.py            loops over settings; one result row per run
│   ├── results_io.py             save/load result rows (CSV, resumable)
│   └── plots.py                 figures and tables read from saved results
│
├── tests/
│   ├── test_projector.py         projector and adjoint tests
│   ├── test_methods.py           FBP, MLEM, OSEM, and MLEM+TV tests
│   └── test_basics.py            phantom, metrics, and scan tests
│
├── reference/
│   └── tomo_step1.py             owner's prototype; calibration anchor only; never modified (D4)
│
├── results/                      saved CSV/figures only; no logic
│   └── .gitkeep                  keeps the empty folder in git (D20)
│
├── pytest.ini                    pythonpath = src, testpaths = tests (D1)
├── .gitignore                    .venv, __pycache__, .pytest_cache (D1)
├── README.md                     project documentation; written in Step 10,
│                                 with numbers only from results/
│
└── requirements.txt              the 5 direct packages, pinned from the real environment (D2)
```

Contracts (names, inputs, outputs):

| Function | Inputs | Output |
|---|---|---|
| `Settings` (config.py) | n=128, n_angles=60, counts=2e5, seed=0, n_iter=100, n_subsets=1, beta=0.0, lesion=False | dataclass |
| `make_fine_phantom(n, lesion=False, lesion_contrast=0.25)` | coarse image size, optional small lesion, optional contrast (default +25%) | `fine_truth[2n,2n]` in [0, 1] (D7, D26) |
| `block_mean(fine)` | `fine[2n,2n]` | `coarse[n,n]`, 2x2 block mean (D7) |
| `make_phantom(n, lesion=False, lesion_contrast=0.25)` | image size, optional small lesion, optional contrast | `(block_mean(make_fine_phantom(n, lesion, lesion_contrast)), mask[n,n])` (D7, D26) |
| `lesion_rois(n)` | image size | `(lesion_roi, background_roi)` boolean masks [n,n]; see D12 |
| `build_operator(n, thetas_deg, n_det=None)` | image size, angles, detector bins (`None` means `n + 2`) | sparse `A`, shape `(n_angles*n_det, n*n)`; fine operator = `build_operator(2n, thetas, n_det=2*(n+2))` (D5, D6) |
| `CountingOperator(A)` | sparse A | object with `.forward(x)`, `.back(y)`, `.n_forward`, `.n_back` |
| `simulate_scan(fine_truth, thetas_deg, counts, seed)` | FINE truth `[2n,2n]`, angles, dose, seed | `(m[n_angles,n+2], scale, expected)` (D8) |
| `fbp(op, m, thetas_deg, window="hann")` | operator, sinogram | image `[n,n]` (data units) |
| `mlem(op, m, n_iter, mask, callback=None)` | operator, sinogram | `(x, history)` |
| `osem(op, m, n_iter, n_subsets, mask, callback=None)` | as above | `(x, history)` |
| `mlem_tv(op, m, n_iter, beta, mask, callback=None)` | as above | `(x, history)`; TV acts on `x/scale` (D16; see open question O1) |
| `score(truth, rec_scaled, mask, lesion_roi, background_roi)` | images in [0,1], boolean masks | dict: psnr, ssim, lesion_contrast, background_noise, contrast_recovery (D9-D11, D25) |
| `run_one(settings, methods)` | settings, method names | list of result-row dicts |
| `save_rows(path, rows)`, `load_rows(path)` | rows | CSV with settings in every row |

## 6. Reference baseline (a calibration anchor, NOT a target)

`reference/tomo_step1.py` is a prototype that uses a slower rotate-and-sum projector. With `n=128, n_angles=60, counts=2e5, seed=0` it gave about: FBP 20.4 dB; MLEM best 22.7 dB at iteration ~20; MLEM at 100 iterations 17.2 dB. With `n_angles=30, counts=3e4` it gave about: FBP 14.6 dB; MLEM best 19.2 dB at iteration ~10; MLEM at 100 iterations 13.5 dB.
Your new projector will differ somewhat. The qualitative pattern must hold (MLEM beats FBP at low counts, then degrades with too many iterations). If your numbers differ from these by more than ~2 dB, investigate and report. NEVER tune anything to match these numbers.

## 7. Build plan (one step at a time; each has a gate)

**Step 0: Setup.** Create the skeleton with a short header comment in each file (what it takes, what it returns). `git init`. Create a venv, install the allowed libraries, pin real versions of the 5 direct packages in `requirements.txt` (D2), record `python --version`. Copy the prototype into `reference/`, run it, and record its real output in `docs/PROGRESS.md`. Gate: `pytest -q` runs (zero tests is fine) and the prototype's output is pasted.

**Step 1: config, phantom, metrics.** Tests: phantom shape/range/mask; `block_mean` of the fine phantom equals the coarse truth; the lesion lies inside a uniform region of the phantom (D7); lesion and background ROIs do not overlap, lie inside the mask, and follow D12; PSNR (mask pixels only) of identical images is infinite and PSNR matches a hand-computed MSE case (D9); SSIM with the D10 settings of identical images is 1; contrast, noise and contrast recovery match hand-computed values on a toy array (D11). Gate: all tests pass.

**Step 2: projector.** Tests (all required): (a) adjoint: for random x, y in float64, `<Ax, y>` equals `<x, Aᵀy>` to relative error below 1e-10; (b) mass conservation: each angle's projection sums to the image sum, using images supported inside the mask (D6); (c) a centered disk gives a symmetric profile, identical for all angles within a measured tolerance; (d) shape and sparsity reported; build time for `n=128, 60 angles` reported. Add `CountingOperator`. Gate: all pass.

**Step 3: scan.** Tests: seed reproducibility (identical `m` twice); `m` is non-negative integers; mean of `m / (scale*expected)` is near 1 over many bins at high counts; total counts within statistical tolerance of the target; the fine and coarse operators are provably different objects. Gate: all pass.

**Step 4: FBP.** Derive the global scale analytically for the exact discretisation used (angles span 180 degrees, detector spacing 1; the starting point is pi / n_angles) and write the derivation in `docs/DECISIONS.md`. Verify it with the noise-free test; NEVER fit it to the truth (D14). Pad each projection to `2*n_det` before the FFT, ramp filter times Hann window, crop, back-project with Aᵀ. NEVER fit a scale to the ground truth. Tests: noise-free scan with 180 angles reconstructs with mean intensity inside the phantom within a measured tolerance, and a measured PSNR floor; more angles never gives a worse noise-free PSNR. Gate: all pass.

**Step 5: MLEM.** Update: `x <- x / s * Aᵀ(m / (A x + eps))`, flat start inside the mask. Tests: non-negativity at every iteration; log-likelihood non-decreasing across iterations (float64); `sum(A x_k)` equals `sum(m)` on data generated by the SAME coarse operator (`m = Poisson(A x)`), within a tolerance set after measuring the real deviation (D13); noise-free data converges toward the truth. Gate: all pass.

**Step 6: OSEM.** Subsets are interleaved angle groups processed in a fixed rotating order; one full pass over all subsets counts as one iteration. Tests: with `n_subsets=1` the result equals MLEM (max abs difference below 1e-10); non-negativity; A/Aᵀ call counts reported per iteration. Gate: all pass.

**Step 7: MLEM + TV (one-step-late).** Update: `x <- x / (s + beta * grad_TV(x)) * Aᵀ(m / (A x + eps))` with smoothed TV (small epsilon inside the square root) applied to `x/scale` (D16), and the denominator floored to stay positive. `beta` is tuned per (angles, counts) on tuning seeds from a small log-spaced grid chosen after measuring (D16). Tests: `beta=0` equals MLEM exactly; analytic TV gradient matches finite differences (relative error below a measured threshold); non-negativity. Gate: all pass.

**Step 8: cost and precision.** Wall-clock time and A/Aᵀ call counts for every method. Precision study: float64 versus float32, and a sinogram rounded to 16-bit and 8-bit integers. Report quality loss. Do NOT claim speed-ups from simulated low precision. Gate: numbers saved to `results/`.

**Step 9: experiments.** Before the sweep, time one run at `n=64` and at `n=128` and estimate the full grid; if it exceeds ~2 hours, propose a reduction and wait for approval (D19). Sweeps: angles [15, 30, 60, 90, 180]; counts [1e4, 3e4, 1e5, 3e5, 1e6]; limited-angle with 3-degree spacing over ranges 90, 120, 150, 180 degrees (30, 40, 50, 60 angles; D18); reporting seeds 0-9, tuning seeds 100-104 (D17); all methods. Each row stores the full settings. Results are resumable and written atomically. Develop at `n=64`, run final at `n=128`. Gate: complete CSVs; a re-run reproduces them exactly.

**Step 10: plots, README, note.** Figures read only from saved CSVs: image grid, error maps, quality vs iterations, vs angles, vs dose, vs compute cost, failure cases. README numbers come from the CSVs (generate them with code). Draft the 2-page note in Markdown (D3). The README notes OSL instability of MLEM+TV (D16). Gate: no hand-typed numbers anywhere.

**Optional (only if I ask):** Streamlit demo; real CT slices via a public dataset (ask me first; check the licence; never download automatically).

## 8. Step report format (mandatory at the end of every step)

```
STEP N REPORT
Files changed:      <list>
Commands run:       <exact commands>
Real output:        <pasted, not paraphrased>
Tests:              <passed/failed counts from pytest -q>
Library signatures verified: <pasted help() lines, or "none new">
Deviations from SPEC: <none, or list and mirror in docs/DECISIONS.md>
UNVERIFIED:         <anything not executed or checked>
Proposals (not done): <ideas, if any>
Ready for approval: yes/no
```

## 9. Integrity rules for results and claims

- **Oracle stopping is labelled.** Choosing the iteration with the best PSNR against the truth is an oracle. Report it separately as "oracle-stopped". Also report a non-oracle rule: the iteration count is chosen per (angles, counts) on tuning seeds and then applied unchanged to the reporting seeds (D15). Never present the oracle number as a realistic result.
- **Tune on separate seeds.** Choose `beta` and iteration counts on seeds that are NOT used for the reported results: tuning seeds 100-104, reporting seeds 0-9 (D17).
- **Same treatment for all methods:** same rescale, same clipping, same data, same seeds.
- **Report spread.** Always mean ± standard deviation over seeds. Flag differences smaller than one standard deviation as ties.
- **State limits plainly** in the README: simulation only; emission-style Poisson model, not transmission CT; parallel-beam; phantom-based; not a clinical tool.
- **No invented references.** Do not cite papers, datasets or benchmarks from memory. Cite only what I give you.
- **No claims about third parties.** Do not state any relationship to organizations, programs or products.

## 10. First action

Do NOT write code yet. Reply with: (1) your understanding of this project in 10 bullets; (2) every assumption you are making; (3) every question you have; (4) anything in this spec that looks inconsistent or risky. Then WAIT for my approval to start Step 0.
