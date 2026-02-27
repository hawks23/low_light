"""Diagnostic script to identify why training loss is stuck."""
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import glob, os

from model import LightweightDarkIR
from losses import DarkIRLoss
from config import Config

device = Config.DEVICE

# Load model
model = LightweightDarkIR().to(device)
model.eval()

criterion = DarkIRLoss(lambda_ssim=Config.LAMBDA_SSIM)

# Load a few real training pairs
low_dir = r"E:\Coding_stuff\Low Light Enhancement\DarkIR_Mod\images\training\low"
high_dir = r"E:\Coding_stuff\Low Light Enhancement\DarkIR_Mod\images\training\normal"
low_files = sorted(glob.glob(os.path.join(low_dir, '*.*')))[:5]
high_files = sorted(glob.glob(os.path.join(high_dir, '*.*')))[:5]

print("="*60)
print("DIAGNOSTIC: Why is training loss stuck?")
print("="*60)

# Test 1: Baseline loss (if model just returned input unchanged)
print("\n--- Test 1: Baseline Loss (dark image vs target) ---")
baseline_l1s = []
for lf, hf in zip(low_files, high_files):
    low = TF.to_tensor(Image.open(lf).convert('RGB')).unsqueeze(0).to(device)
    high = TF.to_tensor(Image.open(hf).convert('RGB')).unsqueeze(0).to(device)
    # Center-crop to same size
    _, _, h1, w1 = low.shape
    _, _, h2, w2 = high.shape
    h = min(h1, h2); w = min(w1, w2)
    h = h - (h % Config.DOWNSAMPLE_FACTOR); w = w - (w % Config.DOWNSAMPLE_FACTOR)
    low = low[:, :, :h, :w]; high = high[:, :, :h, :w]
    
    l1 = (low - high).abs().mean().item()
    baseline_l1s.append(l1)
    print(f"  {os.path.basename(lf)}: L1(dark, target) = {l1:.4f}")
print(f"  Average baseline L1: {sum(baseline_l1s)/len(baseline_l1s):.4f}")

# Test 2: Model output vs input (is the model changing anything?)
print("\n--- Test 2: Does model change the image? ---")
with torch.no_grad():
    for lf, hf in zip(low_files[:3], high_files[:3]):
        low = TF.to_tensor(Image.open(lf).convert('RGB')).unsqueeze(0).to(device)
        high = TF.to_tensor(Image.open(hf).convert('RGB')).unsqueeze(0).to(device)
        h = min(low.shape[2], high.shape[2]); w = min(low.shape[3], high.shape[3])
        h = h - (h % Config.DOWNSAMPLE_FACTOR); w = w - (w % Config.DOWNSAMPLE_FACTOR)
        low = low[:, :, :h, :w]; high = high[:, :, :h, :w]
        
        output = model(low)
        diff_from_input = (output - low).abs().mean().item()
        diff_from_target = (output - high).abs().mean().item()
        input_mean = low.mean().item()
        output_mean = output.mean().item()
        target_mean = high.mean().item()
        print(f"  {os.path.basename(lf)}:")
        print(f"    Input mean:  {input_mean:.4f}")
        print(f"    Output mean: {output_mean:.4f}")  
        print(f"    Target mean: {target_mean:.4f}")
        print(f"    |output - input|:  {diff_from_input:.4f}")
        print(f"    |output - target|: {diff_from_target:.4f}")

# Test 3: Gradient magnitudes per layer
print("\n--- Test 3: Gradient magnitudes ---")
model.train()
low = TF.to_tensor(Image.open(low_files[0]).convert('RGB')).unsqueeze(0).to(device)
high = TF.to_tensor(Image.open(high_files[0]).convert('RGB')).unsqueeze(0).to(device)
h = min(low.shape[2], high.shape[2]); w = min(low.shape[3], high.shape[3])
h = h - (h % Config.DOWNSAMPLE_FACTOR); w = w - (w % Config.DOWNSAMPLE_FACTOR)
low = low[:, :, :h, :w]; high = high[:, :, :h, :w]

output = model(low)
loss, loss_dict = criterion(output, high)
loss.backward()
print(f"  Loss breakdown: {loss_dict}")
for name, param in model.named_parameters():
    if param.grad is not None:
        g = param.grad.abs().mean().item()
        gmax = param.grad.abs().max().item()
        print(f"  {name:45s} | grad_mean={g:.8f}  grad_max={gmax:.6f}  param_norm={param.data.norm():.4f}")
