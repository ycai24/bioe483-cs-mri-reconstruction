# Compressed Sensing MRI Reconstruction with ISTA

Reconstructing MR images from undersampled k-space with the Iterative Soft-Thresholding Algorithm (ISTA) and a wavelet sparsity prior, following Lustig et al. (2007).

**Course:** BIOE 483 – Biomedical Computed Imaging Systems, University of Illinois Urbana-Champaign
**Authors:** Ryan Leuba, Yameng Cai

## Overview

MRI scan time grows with the number of k-space samples acquired. Compressed sensing exploits the fact that MR images are sparse in a wavelet basis, so an image can be recovered from far fewer samples than the Nyquist rate requires. This project implements CS-MRI end to end on a Shepp-Logan phantom:

- variable-density Cartesian undersampling masks at acceleration factors R = 2, 4, 6, 8
- a Daubechies-4 wavelet sparsifying transform
- ISTA reconstruction, compared against zero-filled inverse FFT
- PSNR / SSIM / NRMSE evaluation, plus sweeps over the regularization weight λ and the iteration count

The reconstruction solves

```
minimize  (1/2) ||F_u x − y||₂²  +  λ ||Ψ x||₁
```

where `F_u` is the undersampled Fourier operator, `y` the measured k-space, and `Ψ` the wavelet transform. Each ISTA iteration takes a gradient step on the data-fidelity term and then soft-thresholds the wavelet coefficients.

## Results

Shepp-Logan phantom, 256 × 256, λ = 0.005, 100 iterations (`results/metrics_summary.csv`):

| R | Method | PSNR (dB) | SSIM | NRMSE |
|---|--------|-----------|------|-------|
| 2 | Zero-filled | 23.75 | 0.579 | 0.311 |
| 2 | CS-ISTA | **25.01** | 0.474 | **0.269** |
| 4 | Zero-filled | **20.42** | **0.560** | **0.456** |
| 4 | CS-ISTA | 18.97 | 0.319 | 0.539 |
| 6 | Zero-filled | **19.97** | **0.559** | **0.480** |
| 6 | CS-ISTA | 17.40 | 0.278 | 0.646 |
| 8 | Zero-filled | **19.75** | **0.576** | **0.493** |
| 8 | CS-ISTA | 17.55 | 0.282 | 0.635 |

CS-ISTA improves PSNR and NRMSE over zero-filling at R = 2, but falls behind at higher accelerations, and its SSIM is lower throughout. Wavelet sparsity itself is strong (about 94% of coefficients are near zero), so the remaining gap most likely comes from the plain ISTA setup: a fixed step size, a single λ, and no momentum. Accelerated solvers (FISTA), λ tuned per R, or a different sparsifying transform are the natural next steps.

> **Note on the original submission:** the notebook in `notebooks/` is the version as submitted. Its gradient step used the *magnitude* of the inverse FFT, so every iteration moved the image in the same direction, and CS-ISTA stalled at about 2 dB PSNR. `cs_mri_reconstruction.py` keeps the signed gradient, and `results/` was regenerated with it.

## Repository layout

```
cs_mri_reconstruction.py      main script (runs the full study)
notebooks/
  cs_mri_reconstruction.ipynb original notebook with its rendered outputs
results/                      metrics table and reconstructions written by the script
requirements.txt
```

## Running it

```bash
pip install -r requirements.txt
python cs_mri_reconstruction.py
```

The script needs no external data and finishes in under a minute on a laptop. Figures open in matplotlib windows; run it with `MPLBACKEND=Agg` to skip them. Results are written to `results/` next to the script. The `# %%` markers also let you step through it cell by cell in VS Code or Spyder.

## References

1. Lustig, M., Donoho, D., & Pauly, J. M. (2007). Sparse MRI: The application of compressed sensing for rapid MR imaging. *Magnetic Resonance in Medicine*, 58(6), 1182–1195.
2. Lustig, M., Donoho, D., Santos, J. M., & Pauly, J. M. (2008). Compressed sensing MRI. *IEEE Signal Processing Magazine*, 25(2), 72–82.
3. Candès, E. J., Romberg, J., & Tao, T. (2006). Robust uncertainty principles: Exact signal reconstruction from highly incomplete frequency information. *IEEE Transactions on Information Theory*, 52(2), 489–509.
4. Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM Journal on Imaging Sciences*, 2(1), 183–202.
