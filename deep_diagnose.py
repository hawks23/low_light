"""Deep diagnostic: Why isn't DarkIR_Mod learning? (Writes to file)"""
import sys
import torch
import torch.nn.functional as F
import glob, os, math
from PIL import Image
import torchvision.transforms.functional as TF
from model import LightweightDarkIR
from losses import DarkIRLoss
from config import Config
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_

# Redirect output to file
f = open("diag_output.txt", "w", encoding="utf-8")
def p(msg=""):
    print(msg)
    f.write(msg + "\n")
    f.flush()

device = Config.DEVICE
p(f"Device: {device}")
torch.cuda.empty_cache()

low_dir = r"E:\Coding_stuff\Low Light Enhancement\DarkIR_Mod\images\training\low"
high_dir = r"E:\Coding_stuff\Low Light Enhancement\DarkIR_Mod\images\training\normal"
low_files = sorted(glob.glob(os.path.join(low_dir, '*.*')))
high_files = sorted(glob.glob(os.path.join(high_dir, '*.*')))

# TEST 1
p("\n" + "="*60)
p("TEST 1: Dataset Pairing")
p("="*60)
p(f"Low: {len(low_files)}, High: {len(high_files)}, Match: {len(low_files)==len(high_files)}")
for i in range(min(15, len(low_files))):
    ln = os.path.basename(low_files[i])
    hn = os.path.basename(high_files[i]) if i < len(high_files) else "MISSING"
    match = "OK" if ln == hn else "MISMATCH"
    p(f"  [{i}] {ln:30s} <-> {hn:30s} {match}")

# TEST 2
p("\n" + "="*60)
p("TEST 2: Image Statistics")
p("="*60)
for i in range(min(5, len(low_files))):
    low = TF.to_tensor(Image.open(low_files[i]).convert('RGB'))
    high = TF.to_tensor(Image.open(high_files[i]).convert('RGB'))
    pct_nz = ((low < 0.01).float().mean()*100).item()
    ratio = high.mean().item() / max(low.mean().item(), 1e-6)
    p(f"  {os.path.basename(low_files[i]):20s}: low_mean={low.mean():.4f} high_mean={high.mean():.4f} ratio={ratio:.1f}x near_zero={pct_nz:.1f}%")

# TEST 3
p("\n" + "="*60)
p("TEST 3: Y-channel darkness")
p("="*60)
from yuv_conversion import YUVConversion
converter = YUVConversion()
for i in range(min(3, len(low_files))):
    low = TF.to_tensor(Image.open(low_files[i]).convert('RGB')).unsqueeze(0)
    y_ch, _, _ = converter.rgb_to_yuv(low)
    pct_001 = ((y_ch < 0.01).float().mean()*100).item()
    pct_005 = ((y_ch < 0.05).float().mean()*100).item()
    pct_010 = ((y_ch < 0.10).float().mean()*100).item()
    p(f"  {os.path.basename(low_files[i]):20s}: Y_mean={y_ch.mean():.4f} <0.01:{pct_001:.1f}% <0.05:{pct_005:.1f}% <0.10:{pct_010:.1f}%")

# TEST 4
p("\n" + "="*60)
p("TEST 4: AMP vs FP32 gradients (256x256 crop)")
p("="*60)
torch.cuda.empty_cache()
low_img = Image.open(low_files[0]).convert('RGB')
high_img = Image.open(high_files[0]).convert('RGB')
low_t = TF.to_tensor(low_img)[:, :256, :256].unsqueeze(0).to(device)
high_t = TF.to_tensor(high_img)[:, :256, :256].unsqueeze(0).to(device)

# AMP test
model = LightweightDarkIR().to(device)
model.train()
criterion = DarkIRLoss(lambda_ssim=Config.LAMBDA_SSIM)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
scaler = GradScaler('cuda')

optimizer.zero_grad()
with autocast('cuda'):
    out = model(low_t)
    loss, ld = criterion(out, high_t)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
gn_amp = clip_grad_norm_(model.parameters(), max_norm=1.0)

nan_count = sum(1 for _, pa in model.named_parameters() 
                 if pa.grad is not None and (torch.isnan(pa.grad).any() or torch.isinf(pa.grad).any()))
total_p = sum(1 for _, pa in model.named_parameters() if pa.grad is not None)
p(f"  AMP:  Loss={loss.item():.4f} GradNorm={gn_amp.item()} NaN_grads={nan_count}/{total_p} Scale={scaler.get_scale()}")

if nan_count > 0:
    p("  Params with NaN/Inf grads under AMP:")
    for name, pa in model.named_parameters():
        if pa.grad is not None and (torch.isnan(pa.grad).any() or torch.isinf(pa.grad).any()):
            p(f"    X {name}")

del model, optimizer, scaler
torch.cuda.empty_cache()

# FP32 test
model2 = LightweightDarkIR().to(device)
model2.train()
optimizer2 = torch.optim.AdamW(model2.parameters(), lr=2e-4)

optimizer2.zero_grad()
out2 = model2(low_t)
loss2, ld2 = criterion(out2, high_t)
loss2.backward()
gn_fp32 = clip_grad_norm_(model2.parameters(), max_norm=1.0)

nan_count2 = sum(1 for _, pa in model2.named_parameters() 
                  if pa.grad is not None and (torch.isnan(pa.grad).any() or torch.isinf(pa.grad).any()))
total_p2 = sum(1 for _, pa in model2.named_parameters() if pa.grad is not None)
p(f"  FP32: Loss={loss2.item():.4f} GradNorm={gn_fp32.item():.4f} NaN_grads={nan_count2}/{total_p2}")

p("\n  Per-layer gradients (FP32):")
for name, pa in model2.named_parameters():
    if pa.grad is not None:
        gmean = pa.grad.abs().mean().item()
        gmax = pa.grad.abs().max().item()
        p(f"    {name:45s} | grad_mean={gmean:.8f} grad_max={gmax:.6f}")

del model2, optimizer2
torch.cuda.empty_cache()

# TEST 5
p("\n" + "="*60)
p("TEST 5: Multiplicative vs Additive gradient strength")
p("="*60)
for y_val_f, label in [(0.005, "very dark"), (0.02, "dark"), (0.1, "medium"), (0.3, "moderate")]:
    y_val = torch.tensor(y_val_f, device=device)
    target_val = torch.tensor(0.5, device=device)
    delta_m = torch.tensor(0.5413, requires_grad=True, device=device)
    loss_m = torch.abs(y_val * F.softplus(delta_m) - target_val)
    loss_m.backward()
    delta_a = torch.tensor(0.0, requires_grad=True, device=device)
    loss_a = torch.abs(y_val + delta_a - target_val)
    loss_a.backward()
    ratio = abs(delta_a.grad.item() / max(abs(delta_m.grad.item()), 1e-12))
    p(f"  y={y_val_f:.3f} ({label:10s}): mult_grad={delta_m.grad.item():.6f} add_grad={delta_a.grad.item():.6f} ratio={ratio:.0f}x")

# SUMMARY
p("\n" + "="*60)
p("DIAGNOSIS SUMMARY")
p("="*60)
p("ISSUE 1: GradNorm=nan -> AMP (fp16) + FFT produces NaN gradients")
p("  Fix: Force FFT block to always compute in fp32.")
p("ISSUE 2: Multiplicative Y branch has vanishing gradients for dark pixels")
p("  y_enh = y * softplus(delta): when y ~ 0, gradient ~ y ~ 0")
p("  Fix: Switch to hybrid: y_enh = y * scale + offset")
p("")

f.close()
print("Done! Output written to diag_output.txt")
