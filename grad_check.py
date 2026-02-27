"""Quick gradient flow test after fixes."""
import torch
from model import LightweightDarkIR
from losses import DarkIRLoss
from config import Config
from torch.nn.utils import clip_grad_norm_

f = open("grad_check.txt", "w")
def p(msg=""):
    print(msg)
    f.write(msg + "\n")

device = Config.DEVICE
model = LightweightDarkIR().to(device)
model.train()
criterion = DarkIRLoss(lambda_ssim=Config.LAMBDA_SSIM)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

# Random input (simulating a dark image)
low = torch.rand(2, 3, 256, 256).to(device) * 0.2  # dark
high = torch.rand(2, 3, 256, 256).to(device) * 0.7  # brighter target

optimizer.zero_grad()
# In train mode, it returns (rgb_enh, pred_yuv, input_yuv)
out, pred_yuv, input_yuv = model(low)

# Need target YUV to compute chrominance loss
_, target_u, target_v = model.yuv_converter.rgb_to_yuv(high)
target_yuv = (None, target_u, target_v)

loss, ld = criterion(out, high, pred_yuv=pred_yuv, target_yuv=target_yuv)
loss.backward()
gn = clip_grad_norm_(model.parameters(), max_norm=1.0)

p(f"Loss: {loss.item():.4f}")
p(f"GradNorm: {gn.item():.4f}")
p(f"Loss dict: {ld}")
p("")

zero_count = 0
nan_count = 0
total = 0
for name, pa in model.named_parameters():
    if pa.grad is not None:
        total += 1
        gmean = pa.grad.abs().mean().item()
        gmax = pa.grad.abs().max().item()
        has_nan = torch.isnan(pa.grad).any().item()
        status = "OK"
        if has_nan:
            status = "NaN!"
            nan_count += 1
        elif gmean < 1e-10:
            status = "DEAD"
            zero_count += 1
        p(f"  {status:5s} {name:45s} grad_mean={gmean:.2e} grad_max={gmax:.2e}")

p(f"\nSummary: {total-zero_count-nan_count}/{total} params alive, {zero_count} dead, {nan_count} NaN")
if zero_count == 0 and nan_count == 0:
    p("ALL PARAMS HAVE HEALTHY GRADIENTS!")
f.close()
