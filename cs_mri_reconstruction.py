"""Compressed Sensing MRI Reconstruction with ISTA (BIOE 483 final project).

Converted from the original Jupyter notebook. Cells are separated by
`# %%` markers, so the file can be run top-to-bottom as a script or
cell-by-cell in VS Code / Spyder.
"""


# %% [markdown]
# ## 1. Import Required Libraries


# %%
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
import pywt
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting
plt.rcParams['figure.figsize'] = (15, 5)
plt.rcParams['font.size'] = 10

print("Libraries imported successfully!")


# %% [markdown]
# ## 2. Generate MRI Phantom Data
#
# We'll use the Shepp-Logan phantom, a standard test image for MRI reconstruction algorithms.


# %%
def generate_phantom(size: int = 256) -> np.ndarray:
    """
    Generate a Shepp-Logan phantom of specified size.
    
    Args:
        size: Image dimension (size x size)
    
    Returns:
        2D numpy array containing the phantom image
    """
    phantom = shepp_logan_phantom()
    phantom = resize(phantom, (size, size), anti_aliasing=True)
    return phantom

# Generate phantom
image_size = 256
phantom = generate_phantom(image_size)

# Display the phantom
plt.figure(figsize=(6, 6))
plt.imshow(phantom, cmap='gray')
plt.title('Original Shepp-Logan Phantom')
plt.colorbar()
plt.axis('off')
plt.tight_layout()
plt.show()

print(f"Phantom shape: {phantom.shape}")
print(f"Phantom range: [{phantom.min():.3f}, {phantom.max():.3f}]")


# %% [markdown]
# ## 3. K-Space Simulation
#
# Simulate fully-sampled k-space data by applying 2D Fourier Transform.


# %%
def forward_fft(image: np.ndarray) -> np.ndarray:
    """
    Apply 2D FFT to transform image to k-space.
    
    Args:
        image: 2D image array
    
    Returns:
        k-space data (complex-valued)
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))

def inverse_fft(kspace: np.ndarray) -> np.ndarray:
    """
    Apply 2D IFFT to transform k-space to image.
    
    Args:
        kspace: k-space data (complex-valued)
    
    Returns:
        Reconstructed image
    """
    return np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace))))

# Generate fully-sampled k-space
kspace_full = forward_fft(phantom)

# Visualize k-space magnitude (log scale for better visibility)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(np.log(np.abs(kspace_full) + 1), cmap='hot')
axes[0].set_title('K-Space Magnitude (Log Scale)')
axes[0].axis('off')
axes[0].set_aspect('equal')

axes[1].imshow(np.angle(kspace_full), cmap='twilight')
axes[1].set_title('K-Space Phase')
axes[1].axis('off')
axes[1].set_aspect('equal')

plt.tight_layout()
plt.show()

print(f"K-space shape: {kspace_full.shape}")
print(f"K-space is complex: {np.iscomplexobj(kspace_full)}")


# %% [markdown]
# ## 4. Undersampling Masks
#
# Create variable density Cartesian undersampling masks at different acceleration factors.


# %%
def create_cartesian_mask(shape: Tuple[int, int], 
                         acceleration_factor: float,
                         center_fraction: float = 0.08) -> np.ndarray:
    """
    Create a variable density Cartesian undersampling mask.
    
    Args:
        shape: Image dimensions (height, width)
        acceleration_factor: Undersampling factor (R)
        center_fraction: Fraction of center k-space to fully sample
    
    Returns:
        Binary mask (1 = sampled, 0 = not sampled)
    """
    mask = np.zeros(shape)
    center_lines = int(shape[0] * center_fraction)
    
    # Fully sample center of k-space
    center_start = shape[0] // 2 - center_lines // 2
    center_end = center_start + center_lines
    mask[center_start:center_end, :] = 1
    
    # Random undersampling for outer k-space
    outer_lines = shape[0] - center_lines
    num_outer_samples = int(outer_lines / acceleration_factor)
    
    # Sample outer regions
    outer_indices = []
    for i in range(shape[0]):
        if i < center_start or i >= center_end:
            outer_indices.append(i)
    
    sampled_lines = np.random.choice(outer_indices, 
                                     size=min(num_outer_samples, len(outer_indices)), 
                                     replace=False)
    mask[sampled_lines, :] = 1
    
    actual_acceleration = shape[0] / np.sum(mask[:, 0])
    
    return mask, actual_acceleration

# Create masks for different acceleration factors
acceleration_factors = [2, 4, 6, 8]
masks = {}

fig, axes = plt.subplots(1, len(acceleration_factors), figsize=(15, 4))

for idx, R in enumerate(acceleration_factors):
    mask, actual_R = create_cartesian_mask((image_size, image_size), R)
    masks[R] = mask
    
    axes[idx].imshow(mask, cmap='gray', aspect='auto')
    axes[idx].set_title(f'R = {R}\n(Actual: {actual_R:.2f})')
    axes[idx].axis('off')
    
    sampling_percentage = (np.sum(mask) / mask.size) * 100
    print(f"R={R}: Sampling {sampling_percentage:.1f}% of k-space")

plt.suptitle('Variable Density Cartesian Undersampling Masks', fontsize=14)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 5. Wavelet Transform (Sparsifying Transform)
#
# Implement wavelet transform as the sparsifying operator $\Psi$.


# %%
def wavelet_forward(image: np.ndarray, 
                   wavelet: str = 'db4', 
                   level: int = 4) -> np.ndarray:
    """
    Apply 2D wavelet transform (analysis).
    
    Args:
        image: Input image
        wavelet: Wavelet type (e.g., 'db4', 'haar')
        level: Decomposition level
    
    Returns:
        Wavelet coefficients
    """
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    # Flatten coefficients into 1D array
    arr, slices = pywt.coeffs_to_array(coeffs)
    return arr, slices, coeffs

def wavelet_inverse(coeffs_array: np.ndarray, 
                   slices, 
                   wavelet: str = 'db4') -> np.ndarray:
    """
    Apply inverse 2D wavelet transform (synthesis).
    
    Args:
        coeffs_array: Flattened wavelet coefficients
        slices: Slice information for reconstruction
        wavelet: Wavelet type
    
    Returns:
        Reconstructed image
    """
    coeffs = pywt.array_to_coeffs(coeffs_array, slices, output_format='wavedec2')
    return pywt.waverec2(coeffs, wavelet)

# Test wavelet transform
wavelet_type = 'db4'
decomp_level = 4

coeffs_array, slices, coeffs = wavelet_forward(phantom, wavelet_type, decomp_level)
reconstructed = wavelet_inverse(coeffs_array, slices, wavelet_type)

# Visualize wavelet coefficients
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(phantom, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(np.abs(coeffs_array), cmap='gray')
axes[1].set_title(f'Wavelet Coefficients\n({wavelet_type}, level={decomp_level})')
axes[1].axis('off')

axes[2].imshow(reconstructed, cmap='gray')
axes[2].set_title('Reconstructed from Wavelets')
axes[2].axis('off')

plt.tight_layout()
plt.show()

# Check sparsity
threshold = 0.01 * np.max(np.abs(coeffs_array))
sparse_coeffs = np.sum(np.abs(coeffs_array) > threshold)
total_coeffs = coeffs_array.size
sparsity_ratio = (total_coeffs - sparse_coeffs) / total_coeffs * 100

print(f"Wavelet sparsity: {sparsity_ratio:.1f}% of coefficients are near-zero")
print(f"Reconstruction error: {np.linalg.norm(phantom - reconstructed):.6f}")


# %% [markdown]
# ## 6. Soft Thresholding Operator
#
# The soft thresholding operator is key to promoting sparsity in the wavelet domain.


# %%
def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply soft thresholding operator.
    
    Soft threshold function: S_λ(x) = sign(x) * max(|x| - λ, 0)
    
    Args:
        x: Input array
        threshold: Threshold value (λ)
    
    Returns:
        Soft-thresholded array
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

# Visualize soft thresholding function
x = np.linspace(-5, 5, 1000)
thresholds = [0.5, 1.0, 2.0]

plt.figure(figsize=(10, 6))
plt.plot(x, x, 'k--', label='Identity', linewidth=2)

for thresh in thresholds:
    y = soft_threshold(x, thresh)
    plt.plot(x, y, label=f'λ = {thresh}', linewidth=2)

plt.xlabel('Input value')
plt.ylabel('Output value')
plt.title('Soft Thresholding Function')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 7. Iterative Soft-Thresholding Algorithm (ISTA)
#
# Implement the core reconstruction algorithm based on Lustig et al. (2007).


# %%
def ista_reconstruction(kspace_undersampled: np.ndarray,
                       mask: np.ndarray,
                       lambda_reg: float = 0.01,
                       num_iterations: int = 100,
                       wavelet: str = 'db4',
                       level: int = 4,
                       step_size: float = 1.0,
                       verbose: bool = True) -> Tuple[np.ndarray, list]:
    """
    Iterative Soft-Thresholding Algorithm for CS-MRI reconstruction.
    
    Solves: minimize (1/2)||F_u*x - y||² + λ||Ψ*x||₁
    
    Args:
        kspace_undersampled: Undersampled k-space data
        mask: Sampling mask
        lambda_reg: Regularization parameter (λ)
        num_iterations: Number of iterations
        wavelet: Wavelet type for sparsifying transform
        level: Wavelet decomposition level
        step_size: Step size for gradient descent
        verbose: Print progress
    
    Returns:
        Reconstructed image and list of objective values
    """
    # Initialize with zero-filled reconstruction
    x = inverse_fft(kspace_undersampled)
    
    objective_values = []
    
    for iteration in range(num_iterations):
        # Data consistency step: gradient of data fidelity term
        # ∇f(x) = F_u^H * (F_u*x - y)
        kspace_current = forward_fft(x)
        data_residual = mask * (kspace_current - kspace_undersampled)
        # Keep the sign of the gradient: inverse_fft() returns a magnitude image,
        # which would make every step push x in the same direction.
        gradient = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(data_residual))))
        
        # Gradient descent step
        x_temp = x - step_size * gradient
        
        # Sparsity-promoting step: soft threshold in wavelet domain
        coeffs_array, slices, _ = wavelet_forward(x_temp, wavelet, level)
        coeffs_thresholded = soft_threshold(coeffs_array, lambda_reg * step_size)
        x = wavelet_inverse(coeffs_thresholded, slices, wavelet)
        
        # Compute objective function value
        kspace_current = forward_fft(x)
        data_fidelity = 0.5 * np.linalg.norm(mask * (kspace_current - kspace_undersampled))**2
        sparsity_term = lambda_reg * np.sum(np.abs(coeffs_thresholded))
        objective = data_fidelity + sparsity_term
        objective_values.append(objective)
        
        if verbose and (iteration + 1) % 20 == 0:
            print(f"Iteration {iteration + 1}/{num_iterations}: Objective = {objective:.4e}")
    
    return x, objective_values

print("ISTA reconstruction function defined successfully!")


# %% [markdown]
# ## 8. Zero-Filled Reconstruction (Baseline)
#
# Create baseline reconstructions using simple inverse FFT for comparison.


# %%
def zero_filled_reconstruction(kspace_undersampled: np.ndarray) -> np.ndarray:
    """
    Perform zero-filled reconstruction (simple inverse FFT).
    
    Args:
        kspace_undersampled: Undersampled k-space data
    
    Returns:
        Reconstructed image
    """
    return inverse_fft(kspace_undersampled)

print("Zero-filled reconstruction function defined!")


# %% [markdown]
# ## 9. Image Quality Metrics
#
# Implement quantitative metrics to evaluate reconstruction quality.


# %%
def compute_metrics(reference: np.ndarray, 
                   reconstructed: np.ndarray) -> dict:
    """
    Compute image quality metrics.
    
    Args:
        reference: Ground truth image
        reconstructed: Reconstructed image
    
    Returns:
        Dictionary containing PSNR, SSIM, and NRMSE
    """
    # Normalize images to same range
    ref_norm = (reference - reference.min()) / (reference.max() - reference.min())
    rec_norm = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min())
    
    # PSNR: Peak Signal-to-Noise Ratio
    psnr_value = psnr(ref_norm, rec_norm, data_range=1.0)
    
    # SSIM: Structural Similarity Index
    ssim_value = ssim(ref_norm, rec_norm, data_range=1.0)
    
    # NRMSE: Normalized Root Mean Square Error
    rmse = np.sqrt(np.mean((ref_norm - rec_norm)**2))
    nrmse = rmse / np.std(ref_norm)
    
    return {
        'PSNR': psnr_value,
        'SSIM': ssim_value,
        'NRMSE': nrmse
    }

print("Image quality metrics function defined!")


# %% [markdown]
# ## 10. Reconstruction Experiments
#
# Perform CS-MRI reconstruction at different acceleration factors and compare with zero-filled reconstruction.


# %%
# Reconstruction parameters
lambda_reg = 0.005
num_iterations = 100
wavelet_type = 'db4'
decomp_level = 4

# Storage for results
results = {
    'zero_filled': {},
    'cs_recon': {},
    'metrics_zf': {},
    'metrics_cs': {},
    'objectives': {}
}

print("Starting reconstructions for different acceleration factors...\n")

for R in acceleration_factors:
    print(f"\n{'='*60}")
    print(f"Acceleration Factor R = {R}")
    print(f"{'='*60}\n")
    
    # Get undersampled k-space
    mask = masks[R]
    kspace_undersampled = kspace_full * mask
    
    # Zero-filled reconstruction
    print("Performing zero-filled reconstruction...")
    recon_zf = zero_filled_reconstruction(kspace_undersampled)
    results['zero_filled'][R] = recon_zf
    
    # CS reconstruction using ISTA
    print(f"\nPerforming CS-ISTA reconstruction (λ={lambda_reg}, iterations={num_iterations})...")
    recon_cs, objectives = ista_reconstruction(
        kspace_undersampled,
        mask,
        lambda_reg=lambda_reg,
        num_iterations=num_iterations,
        wavelet=wavelet_type,
        level=decomp_level,
        verbose=True
    )
    results['cs_recon'][R] = recon_cs
    results['objectives'][R] = objectives
    
    # Compute metrics
    print("\nComputing quality metrics...")
    metrics_zf = compute_metrics(phantom, recon_zf)
    metrics_cs = compute_metrics(phantom, recon_cs)
    
    results['metrics_zf'][R] = metrics_zf
    results['metrics_cs'][R] = metrics_cs
    
    print(f"\nZero-Filled: PSNR={metrics_zf['PSNR']:.2f} dB, SSIM={metrics_zf['SSIM']:.4f}, NRMSE={metrics_zf['NRMSE']:.4f}")
    print(f"CS-ISTA:     PSNR={metrics_cs['PSNR']:.2f} dB, SSIM={metrics_cs['SSIM']:.4f}, NRMSE={metrics_cs['NRMSE']:.4f}")
    print(f"\nImprovement: PSNR={metrics_cs['PSNR']-metrics_zf['PSNR']:.2f} dB, SSIM={metrics_cs['SSIM']-metrics_zf['SSIM']:.4f}")

print("\n" + "="*60)
print("All reconstructions completed!")
print("="*60)


# %% [markdown]
# ## 11. Visualization: Reconstructed Images and Error Maps


# %%
# Visualize reconstructions for all acceleration factors
fig, axes = plt.subplots(len(acceleration_factors), 4, figsize=(16, 4*len(acceleration_factors)))

for idx, R in enumerate(acceleration_factors):
    recon_zf = results['zero_filled'][R]
    recon_cs = results['cs_recon'][R]
    
    # Normalize for display
    phantom_norm = (phantom - phantom.min()) / (phantom.max() - phantom.min())
    recon_zf_norm = (recon_zf - recon_zf.min()) / (recon_zf.max() - recon_zf.min())
    recon_cs_norm = (recon_cs - recon_cs.min()) / (recon_cs.max() - recon_cs.min())
    
    # Error maps
    error_zf = np.abs(phantom_norm - recon_zf_norm)
    error_cs = np.abs(phantom_norm - recon_cs_norm)
    
    # Plot zero-filled reconstruction
    axes[idx, 0].imshow(recon_zf_norm, cmap='gray', vmin=0, vmax=1)
    axes[idx, 0].set_title(f'Zero-Filled (R={R})\nPSNR: {results["metrics_zf"][R]["PSNR"]:.2f} dB')
    axes[idx, 0].axis('off')
    
    # Plot CS reconstruction
    axes[idx, 1].imshow(recon_cs_norm, cmap='gray', vmin=0, vmax=1)
    axes[idx, 1].set_title(f'CS-ISTA (R={R})\nPSNR: {results["metrics_cs"][R]["PSNR"]:.2f} dB')
    axes[idx, 1].axis('off')
    
    # Plot error maps
    im1 = axes[idx, 2].imshow(error_zf, cmap='hot', vmin=0, vmax=0.3)
    axes[idx, 2].set_title(f'Zero-Filled Error')
    axes[idx, 2].axis('off')
    plt.colorbar(im1, ax=axes[idx, 2], fraction=0.046)
    
    im2 = axes[idx, 3].imshow(error_cs, cmap='hot', vmin=0, vmax=0.3)
    axes[idx, 3].set_title(f'CS-ISTA Error')
    axes[idx, 3].axis('off')
    plt.colorbar(im2, ax=axes[idx, 3], fraction=0.046)

plt.suptitle('Reconstruction Comparison: Zero-Filled vs CS-ISTA', fontsize=16, y=0.995)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 12. Convergence Analysis


# %%
# Plot convergence curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Objective function values
for R in acceleration_factors:
    objectives = results['objectives'][R]
    axes[0].plot(objectives, label=f'R={R}', linewidth=2)

axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Objective Function Value')
axes[0].set_title('ISTA Convergence')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')

# Objective function (zoomed in)
for R in acceleration_factors:
    objectives = results['objectives'][R]
    axes[1].plot(range(20, num_iterations), objectives[20:], label=f'R={R}', linewidth=2)

axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Objective Function Value')
axes[1].set_title('ISTA Convergence (Iterations 20-100)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# %% [markdown]
# ## 13. Quantitative Results Summary


# %%
# Create summary table
import pandas as pd

summary_data = []

for R in acceleration_factors:
    metrics_zf = results['metrics_zf'][R]
    metrics_cs = results['metrics_cs'][R]
    
    summary_data.append({
        'Acceleration Factor': R,
        'Method': 'Zero-Filled',
        'PSNR (dB)': f"{metrics_zf['PSNR']:.2f}",
        'SSIM': f"{metrics_zf['SSIM']:.4f}",
        'NRMSE': f"{metrics_zf['NRMSE']:.4f}"
    })
    
    summary_data.append({
        'Acceleration Factor': R,
        'Method': 'CS-ISTA',
        'PSNR (dB)': f"{metrics_cs['PSNR']:.2f}",
        'SSIM': f"{metrics_cs['SSIM']:.4f}",
        'NRMSE': f"{metrics_cs['NRMSE']:.4f}"
    })

df_summary = pd.DataFrame(summary_data)
print("\n" + "="*70)
print("QUANTITATIVE RESULTS SUMMARY")
print("="*70)
print(df_summary.to_string(index=False))
print("="*70)


# %% [markdown]
# ## 14. Performance Comparison Plots


# %%
# Extract metrics for plotting
psnr_zf = [results['metrics_zf'][R]['PSNR'] for R in acceleration_factors]
psnr_cs = [results['metrics_cs'][R]['PSNR'] for R in acceleration_factors]
ssim_zf = [results['metrics_zf'][R]['SSIM'] for R in acceleration_factors]
ssim_cs = [results['metrics_cs'][R]['SSIM'] for R in acceleration_factors]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PSNR comparison
x_pos = np.arange(len(acceleration_factors))
width = 0.35

axes[0].bar(x_pos - width/2, psnr_zf, width, label='Zero-Filled', alpha=0.8)
axes[0].bar(x_pos + width/2, psnr_cs, width, label='CS-ISTA', alpha=0.8)
axes[0].set_xlabel('Acceleration Factor (R)')
axes[0].set_ylabel('PSNR (dB)')
axes[0].set_title('Peak Signal-to-Noise Ratio Comparison')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(acceleration_factors)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# SSIM comparison
axes[1].bar(x_pos - width/2, ssim_zf, width, label='Zero-Filled', alpha=0.8)
axes[1].bar(x_pos + width/2, ssim_cs, width, label='CS-ISTA', alpha=0.8)
axes[1].set_xlabel('Acceleration Factor (R)')
axes[1].set_ylabel('SSIM')
axes[1].set_title('Structural Similarity Index Comparison')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(acceleration_factors)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Improvement analysis
psnr_improvement = [psnr_cs[i] - psnr_zf[i] for i in range(len(acceleration_factors))]
ssim_improvement = [ssim_cs[i] - ssim_zf[i] for i in range(len(acceleration_factors))]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(acceleration_factors, psnr_improvement, 'o-', linewidth=2, markersize=8)
axes[0].set_xlabel('Acceleration Factor (R)')
axes[0].set_ylabel('PSNR Improvement (dB)')
axes[0].set_title('CS-ISTA PSNR Improvement over Zero-Filled')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color='k', linestyle='--', linewidth=1)

axes[1].plot(acceleration_factors, ssim_improvement, 'o-', linewidth=2, markersize=8, color='orange')
axes[1].set_xlabel('Acceleration Factor (R)')
axes[1].set_ylabel('SSIM Improvement')
axes[1].set_title('CS-ISTA SSIM Improvement over Zero-Filled')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color='k', linestyle='--', linewidth=1)

plt.tight_layout()
plt.show()


# %% [markdown]
# ## 15. Parameter Sensitivity Analysis
#
# Investigate the effect of regularization parameter λ on reconstruction quality.


# %%
# Test different lambda values at R=4
test_R = 4
lambda_values = [0.001, 0.005, 0.01, 0.02, 0.05]

lambda_results = {'psnr': [], 'ssim': [], 'recons': []}

print(f"Testing regularization parameter sensitivity at R={test_R}...\n")

mask = masks[test_R]
kspace_undersampled = kspace_full * mask

for lambda_val in lambda_values:
    print(f"λ = {lambda_val}...")
    recon, _ = ista_reconstruction(
        kspace_undersampled,
        mask,
        lambda_reg=lambda_val,
        num_iterations=100,
        wavelet=wavelet_type,
        level=decomp_level,
        verbose=False
    )
    
    metrics = compute_metrics(phantom, recon)
    lambda_results['psnr'].append(metrics['PSNR'])
    lambda_results['ssim'].append(metrics['SSIM'])
    lambda_results['recons'].append(recon)
    
    print(f"  PSNR: {metrics['PSNR']:.2f} dB, SSIM: {metrics['SSIM']:.4f}\n")

# Plot sensitivity analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(lambda_values, lambda_results['psnr'], 'o-', linewidth=2, markersize=8)
axes[0].set_xlabel('Regularization Parameter (λ)')
axes[0].set_ylabel('PSNR (dB)')
axes[0].set_title(f'PSNR vs λ (R={test_R})')
axes[0].set_xscale('log')
axes[0].grid(True, alpha=0.3)

axes[1].plot(lambda_values, lambda_results['ssim'], 'o-', linewidth=2, markersize=8, color='orange')
axes[1].set_xlabel('Regularization Parameter (λ)')
axes[1].set_ylabel('SSIM')
axes[1].set_title(f'SSIM vs λ (R={test_R})')
axes[1].set_xscale('log')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Find optimal lambda
optimal_idx = np.argmax(lambda_results['psnr'])
optimal_lambda = lambda_values[optimal_idx]
print(f"\nOptimal λ = {optimal_lambda} (PSNR: {lambda_results['psnr'][optimal_idx]:.2f} dB)")


# %% [markdown]
# ## 16. Iteration Count Analysis
#
# Study the effect of number of iterations on reconstruction quality.


# %%
# Test different iteration counts at R=4
test_R = 4
iteration_counts = [10, 20, 50, 100, 200]

iteration_results = {'psnr': [], 'ssim': [], 'time': []}

print(f"Testing iteration count sensitivity at R={test_R}, λ={lambda_reg}...\n")

mask = masks[test_R]
kspace_undersampled = kspace_full * mask

import time

for num_iters in iteration_counts:
    print(f"Iterations = {num_iters}...")
    
    start_time = time.time()
    recon, _ = ista_reconstruction(
        kspace_undersampled,
        mask,
        lambda_reg=lambda_reg,
        num_iterations=num_iters,
        wavelet=wavelet_type,
        level=decomp_level,
        verbose=False
    )
    elapsed_time = time.time() - start_time
    
    metrics = compute_metrics(phantom, recon)
    iteration_results['psnr'].append(metrics['PSNR'])
    iteration_results['ssim'].append(metrics['SSIM'])
    iteration_results['time'].append(elapsed_time)
    
    print(f"  PSNR: {metrics['PSNR']:.2f} dB, SSIM: {metrics['SSIM']:.4f}, Time: {elapsed_time:.2f}s\n")

# Plot iteration analysis
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(iteration_counts, iteration_results['psnr'], 'o-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Iterations')
axes[0].set_ylabel('PSNR (dB)')
axes[0].set_title(f'PSNR vs Iteration Count (R={test_R})')
axes[0].grid(True, alpha=0.3)

axes[1].plot(iteration_counts, iteration_results['ssim'], 'o-', linewidth=2, markersize=8, color='orange')
axes[1].set_xlabel('Number of Iterations')
axes[1].set_ylabel('SSIM')
axes[1].set_title(f'SSIM vs Iteration Count (R={test_R})')
axes[1].grid(True, alpha=0.3)

axes[2].plot(iteration_counts, iteration_results['time'], 'o-', linewidth=2, markersize=8, color='green')
axes[2].set_xlabel('Number of Iterations')
axes[2].set_ylabel('Computation Time (s)')
axes[2].set_title(f'Computation Time vs Iteration Count (R={test_R})')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# %% [markdown]
# ## 17. Discussion and Conclusions
#
# ### Key Findings:
#
# 1. **CS-ISTA Improvement**: The compressed sensing reconstruction using iterative soft-thresholding consistently outperforms zero-filled reconstruction across all acceleration factors, demonstrating significant improvements in both PSNR and SSIM.
#
# 2. **Acceleration vs Quality Trade-off**: As the acceleration factor increases, image quality degrades for both methods, but CS-ISTA maintains superior performance, especially at higher acceleration factors (R=6, R=8).
#
# 3. **Sparsity in Wavelet Domain**: The wavelet transform effectively provides a sparse representation of MR images, with >90% of coefficients being near-zero, which is crucial for CS reconstruction.
#
# 4. **Parameter Sensitivity**:
#    - The regularization parameter λ significantly affects reconstruction quality
#    - Optimal λ balances data fidelity and sparsity promotion
#    - Too small λ leads to insufficient denoising; too large λ over-smooths the image
#
# 5. **Convergence Behavior**: ISTA converges rapidly in the first 20-30 iterations, with diminishing returns after 100 iterations for most acceleration factors.
#
# ### Clinical Implications:
#
# - **Scan Time Reduction**: CS-MRI can reduce scan times by factors of 2-4× while maintaining diagnostic image quality
# - **Patient Comfort**: Shorter scans improve patient compliance and reduce motion artifacts
# - **Throughput**: Enables more patients to be scanned per day
#
# ### Limitations:
#
# 1. Computational complexity: ISTA requires iterative optimization (minutes for high-resolution images)
# 2. Parameter tuning: λ must be carefully chosen for each application
# 3. Phantom data: Real clinical data has more complex noise and artifacts
#
# ### Future Directions:
#
# 1. **Deep Learning Methods**: Compare with recent deep learning-based CS-MRI methods (e.g., unrolled networks)
# 2. **Multi-coil Reconstruction**: Extend to parallel imaging (SENSE, GRAPPA) combined with CS
# 3. **Real Data Validation**: Test on actual clinical MRI datasets
# 4. **Acceleration Optimization**: Investigate optimal undersampling patterns (radial, spiral)
# 5. **Hardware Acceleration**: Implement GPU-accelerated reconstruction for real-time performance


# %% [markdown]
# ## 18. Save Results
#
# Save reconstructed images and metrics for later analysis or presentation.


# %%
import pickle

# Create results directory next to this script
RESULTS_DIR = Path(__file__).resolve().parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# Save results dictionary
with open(RESULTS_DIR / 'reconstruction_results.pkl', 'wb') as f:
    pickle.dump(results, f)

# Save summary table
df_summary.to_csv(RESULTS_DIR / 'metrics_summary.csv', index=False)

# Save individual reconstructions as numpy arrays
np.save(RESULTS_DIR / 'phantom.npy', phantom)
for R in acceleration_factors:
    np.save(RESULTS_DIR / f'recon_zf_R{R}.npy', results['zero_filled'][R])
    np.save(RESULTS_DIR / f'recon_cs_R{R}.npy', results['cs_recon'][R])

print("Results saved successfully!")
print(f"Files saved in {RESULTS_DIR}:")
print("  - reconstruction_results.pkl")
print("  - metrics_summary.csv")
print("  - phantom.npy")
for R in acceleration_factors:
    print(f"  - recon_zf_R{R}.npy, recon_cs_R{R}.npy")


# %% [markdown]
# ## References
#
# 1. Lustig, M., Donoho, D., & Pauly, J. M. (2007). Sparse MRI: The application of compressed sensing for rapid MR imaging. *Magnetic Resonance in Medicine*, 58(6), 1182-1195.
#
# 2. Lustig, M., Donoho, D., Santos, J. M., & Pauly, J. M. (2008). Compressed sensing MRI. *IEEE Signal Processing Magazine*, 25(2), 72-82.
#
# 3. Candès, E. J., Romberg, J., & Tao, T. (2006). Robust uncertainty principles: Exact signal reconstruction from highly incomplete frequency information. *IEEE Transactions on Information Theory*, 52(2), 489-509.
#
# 4. Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM Journal on Imaging Sciences*, 2(1), 183-202.
#
# 5. BIOE483 Course Notes: Biomedical Computed Imaging Systems
