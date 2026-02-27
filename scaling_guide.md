# High-End GPU Training & Scaling Guide

If you are moving this codebase from a local/testing GPU to a massive compute cluster (like an A100 or H100), the current `LightweightDarkIR` configuration is highly bottlenecked by its intentional constraints (< 80K parameters, FP16 optimizations, batch size 4). 

To extract maximum quality and utilize a faster GPU, here are the exact adjustments you need to make across the codebase.

---

## 1. Scale Up the Model Capacity (Hyperparameters)
Currently, the model relies on aggressively small channel counts to stay under 80K parameters. A larger GPU can handle a much deeper and wider feature space.

**File:** `config.py`
Change the following:
* `C = 16` $\rightarrow$ `C = 64` or `128` (Widens the frequency modulation space).
* `UV_CHANNELS = 8` $\rightarrow$ `UV_CHANNELS = 32` or `64` (Allows for far better color reconstruction and stabilization).
* `DOWNSAMPLE_FACTOR = 4` $\rightarrow$ `DOWNSAMPLE_FACTOR = 2` or `1` (Processing the FFT at a higher resolution captures much finer illumination details, though it squares the memory requirement).
* `FFT_BLOCKS = 1` $\rightarrow$ If needed, you can stack multiple `FFTBlock` sequential layers inside `y_branch.py` to allow deeper frequency reasoning.

## 2. Increase Precision Requirements (FP32 or BF16)
Mixed precision (`autocast`) using `FP16` is great for consumer cards, but can sometimes cause vanishing gradients or underflow in frequency operations (hence the complex PyTorch bugs we encountered). 

**File:** `train.py` & `inference.py`
* **Disable AMP constraints**: If memory is no object, you can remove `with autocast():` and the `GradScaler()` logic entirely to train strictly in `Float32`. This guarantees the highest mathematical precision for the FFT blocks.
* **Alternative**: Use `BFloat16` (`bf16`). Modern GPUs (Ampere architectures and newer) natively support `BFloat16` which has the same dynamic range as `Float32` but takes half the memory. Change `autocast()` to `autocast(dtype=torch.bfloat16)`.

## 3. Maximize Data Throughput (Dataloading)
The current script is simple and may bottleneck the GPU by starving it of data.

**File:** `train.py`
* `BATCH_SIZE = 4` $\rightarrow$ `BATCH_SIZE = 16`, `32`, or `64`. (Always push this as high as your VRAM allows to get stable gradients).
* `num_workers=4` $\rightarrow$ `num_workers=8` or `16`. Ensure your CPU can read and decode the `.png` files fast enough.
* `PATCH_SIZE = 512` $\rightarrow$ `PATCH_SIZE = 1024` or even full resolution `2160x3840`. Training on larger patches reduces edge artifacts.

## 4. Enhanced Augmentations
If you scale the model up (increasing parameters), it will be prone to overfitting unless you give it much harder data. 

**File:** `train.py` (Inside `RealLowLightDataset`)
Currently, you only apply `RandomCrop`. You should aggressively expand this:
```python
self.transform_pair = transforms.Compose([
    transforms.RandomCrop(1024),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    # Add random 90-degree rotations
])

# Use torchvision v2 transforms or albumentations to ensure both 
# the low-light and target image receive the EXACT same random flips.
```

## 5. Switch to Distributed Data Parallel (DDP)
If your high-end machine has **multiple GPUs** (e.g., 4x A100s), standard PyTorch will only use `cuda:0`.
* You will need to wrap the model in `torch.nn.parallel.DistributedDataParallel` (DDP).
* Modify the dataloader to use `DistributedSampler` so each GPU gets a unique slice of the dataset simultaneously.

---

### Suggested "Premium" Config for A100/H100 GPUs:
```python
# config.py
DOWNSAMPLE_FACTOR = 2
C = 64
UV_CHANNELS = 32
REDUCTION_RATIO = 2

PATCH_SIZE = 1024
LEARNING_RATE = 1e-4 # Slightly lower if batch size is massive
BATCH_SIZE = 32
```
If you make these changes, the model will transition from a lightweight edge device architecture to a heavily-parameterized studio-grade enhancer.
