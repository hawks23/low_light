# Phase 10: Ablation Testing Plan

This document provides a structured guide on how to perform ablation studies on the `LightweightDarkIR` model to isolate and understand the contribution of its individual components.

## Goals
1. Validate the necessity of the FFT Block for luminance modulation.
2. Asses the impact of the UV chroma stabilization branch.
3. Determine the effectiveness of the channel gating mechanism vs full frequency map scaling.
4. Evaluate the benefit of depthwise spatial refinement before FFT.

## Ablation Configurations

To run these ablations, you will modify `model.py`, `y_branch.py`, or `fft_block.py` and run the training/inference scripts to compare metrics (PSNR, SSIM, Runtime, FLOPs).

### 1. Baseline Model
- **Configuration**: Standard `LightweightDarkIR` (C=16, UV_CHANNELS=8).
- **Expected Results**: Best PSNR/SSIM, low runtime (< 50ms for 4K on modern GPU).

### 2. Removing FFT Modulation
- **Change**: In `y_branch.py`, comment out `feat = self.fft_block(feat)`.
- **Purpose**: Tests if frequency domain modulation is necessary for illumination correction compared to purely spatial convolution.
- **Expected Results**: Significant drop in global contrast enhancement capability; faster runtime.

### 3. Removing UV Branch
- **Change**: In `model.py`, skip `self.uv_branch`. Pass original `u` and `v` directly to `yuv_to_rgb(y_enh, u, v)`.
- **Purpose**: Tests the impact of chroma stabilization.
- **Expected Results**: Noticeable color shift or oversaturation in enhanced images; slight reduction in FLOPs.

### 4. Removing Channel Gating (Using Full Map Scaling)
- **Change**: Modify `fft_block.py`. Instead of pooling energy and using an MLP (`self.mlp(energy)`), use a learnable map of size `[1, C, H_d, W_d//2 + 1]` and multiply it directly with `amplitude`.
- **Purpose**: Tests if the lightweight channel-wise scalar gating is sufficient compared to dense coordinate-wise scaling.
- **Expected Results**: Increased parameter count, potentially marginal PSNR gain, but high risk of overfitting and slower runtime.

### 5. Removing Depthwise Spatial Refinement
- **Change**: In `y_branch.py`, remove `self.dw_conv` step.
- **Purpose**: Tests if local spatial smoothing before the global FFT operation is needed.
- **Expected Results**: Potential introduction of high-frequency artifacts or ringing in the output; very minor speedup.

## Evaluation Process
For each configuration, run the following command to track metrics:
```bash
python inference.py --input test_4k_image.png --target gt_4k_image.png --metrics
```
Compare the output PSNR, SSIM, and MACs/Params against the baseline.
