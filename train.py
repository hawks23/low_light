import os
import csv
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
import argparse
from tqdm import tqdm
import time
import math

import glob
from torchvision import transforms
from PIL import Image

import torchvision.transforms.functional as TF

# Project imports
from config import Config
from model import LightweightDarkIR
from losses import DarkIRLoss


# ─── Metrics ────────────────────────────────────────────────────
def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Compute PSNR in dB between pred and target tensors."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(max_val ** 2 / mse)


# ─── Datasets ───────────────────────────────────────────────────
class RealLowLightDataset(Dataset):
    def __init__(self, low_dir, high_dir, patch_size=Config.PATCH_SIZE):
        self.low_images = sorted(glob.glob(os.path.join(low_dir, '*.*')))
        self.high_images = sorted(glob.glob(os.path.join(high_dir, '*.*')))
        self.patch_size = patch_size
        
        self.transform = transforms.Compose([
            transforms.RandomCrop(patch_size, pad_if_needed=True),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low_img = Image.open(self.low_images[idx]).convert('RGB')
        high_img = Image.open(self.high_images[idx]).convert('RGB')
        
        seed = torch.random.seed()
        torch.manual_seed(seed)
        low_tensor = self.transform(low_img)
        
        torch.manual_seed(seed)
        high_tensor = self.transform(high_img)
        
        if random.random() > 0.5:
            low_tensor = TF.hflip(low_tensor)
            high_tensor = TF.hflip(high_tensor)
        if random.random() > 0.5:
            low_tensor = TF.vflip(low_tensor)
            high_tensor = TF.vflip(high_tensor)
        
        return low_tensor, high_tensor


class ValidationDataset(Dataset):
    """Loads low-light images with optional paired GT for metric validation."""
    def __init__(self, low_dir, high_dir=None):
        self.low_images = sorted(glob.glob(os.path.join(low_dir, '*.*')))
        self.has_gt = high_dir is not None and os.path.exists(high_dir)
        if self.has_gt:
            self.high_images = sorted(glob.glob(os.path.join(high_dir, '*.*')))
            assert len(self.low_images) == len(self.high_images), \
                f"Mismatch: {len(self.low_images)} low vs {len(self.high_images)} high images"
        
    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        img_path = self.low_images[idx]
        low_img = Image.open(img_path).convert('RGB')
        
        w, h = low_img.size
        new_w = w - (w % Config.DOWNSAMPLE_FACTOR)
        new_h = h - (h % Config.DOWNSAMPLE_FACTOR)
        if new_w != w or new_h != h:
            low_img = low_img.resize((new_w, new_h), Image.BILINEAR)

        low_tensor = TF.to_tensor(low_img)
        filename = os.path.basename(img_path)
        
        if self.has_gt:
            high_img = Image.open(self.high_images[idx]).convert('RGB')
            high_img = high_img.resize((new_w, new_h), Image.BILINEAR)
            high_tensor = TF.to_tensor(high_img)
            return low_tensor, high_tensor, filename
        
        return low_tensor, filename


# ─── LR Schedule ────────────────────────────────────────────────
def get_lr_cosine_schedule(epoch, max_epochs, max_lr, min_lr=1e-6, warmup_epochs=10):
    """Cosine Annealing LR with linear warmup."""
    if epoch < warmup_epochs:
        lr = min_lr + (max_lr - min_lr) * (epoch / warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
    return lr


# ─── Training ───────────────────────────────────────────────────
def train(args):
    device = Config.DEVICE
    print(f"Using device: {device}")
    
    # ── Init Model ──
    model = LightweightDarkIR(
        c=Config.C, 
        downsample_factor=Config.DOWNSAMPLE_FACTOR,
        uv_channels=Config.UV_CHANNELS,
        reduction_ratio=Config.REDUCTION_RATIO
    ).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")
    
    # ── Init Loss ──
    criterion = DarkIRLoss(lambda_ssim=Config.LAMBDA_SSIM)
    
    # ── Init Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # ── Dataloaders ──
    train_dataset = RealLowLightDataset(
        low_dir=args.train_low, 
        high_dir=args.train_normal, 
        patch_size=Config.PATCH_SIZE
    )
    
    val_dataset = None
    val_loader = None
    if args.val_low and os.path.exists(args.val_low):
        val_high_dir = args.val_normal if args.val_normal else None
        val_dataset = ValidationDataset(args.val_low, high_dir=val_high_dir)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        os.makedirs(os.path.join(args.ckpt_dir, "val_outputs"), exist_ok=True)
        gt_status = "with GT pairs" if val_dataset.has_gt else "visual only (no GT)"
        print(f"Validation enabled: {len(val_dataset)} images ({gt_status})")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=True
    )
    
    # ── Checkpoint dir ──
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    # ── CSV Log Setup ──
    csv_path = os.path.join(args.ckpt_dir, "training_log.csv")
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_header = [
        'epoch', 'lr', 'train_loss', 'train_l1', 'train_ssim', 'train_freq',
        'train_psnr', 'grad_norm', 'loss_delta', 'val_psnr', 'val_ssim'
    ]
    csv_writer.writerow(csv_header)
    
    # ── Tracking State ──
    best_loss = float('inf')
    prev_loss = None
    stagnation_count = 0
    
    # ── Identity Sanity Check (Epoch 0) ──
    print("\n" + "=" * 60)
    print("INIT SANITY CHECK: Model output vs input at initialization")
    print("=" * 60)
    model.eval()
    with torch.no_grad():
        sample_low, sample_high = train_dataset[0]
        sample_low = sample_low.unsqueeze(0).to(device)
        sample_high = sample_high.unsqueeze(0).to(device)
        
        # In eval mode, model only returns rgb_enh
        init_out = model(sample_low)
        
        init_diff = (init_out - sample_low).abs().mean().item()
        init_psnr_vs_input = compute_psnr(init_out, sample_low)
        init_psnr_vs_target = compute_psnr(init_out, sample_high)
        baseline_psnr = compute_psnr(sample_low, sample_high)
        print(f"  |output - input| mean:    {init_diff:.6f} (should be < 0.05)")
        print(f"  PSNR(output, input):      {init_psnr_vs_input:.2f} dB (should be > 30)")
        print(f"  PSNR(output, target):     {init_psnr_vs_target:.2f} dB")
        print(f"  PSNR(input, target):      {baseline_psnr:.2f} dB (baseline)")
        if init_diff > 0.05:
            print("  WARNING: Init output differs significantly from input!")
        else:
            print("  PASS: Model starts near-identity")
    print("=" * 60 + "\n")

    # ── Gradient Flow Check (first batch) ──
    print("GRADIENT FLOW CHECK (first batch, fp32):")
    model.train()
    check_low, check_high = next(iter(train_loader))
    check_low, check_high = check_low.to(device), check_high.to(device)
    optimizer.zero_grad()
    
    # In train mode, it returns (rgb_enh, pred_yuv, input_yuv)
    check_out, check_pred_yuv, _ = model(check_low)
    _, target_u_chk, target_v_chk = model.yuv_converter.rgb_to_yuv(check_high)
    check_target_yuv = (None, target_u_chk, target_v_chk)
    
    check_loss, _ = criterion(check_out, check_high, pred_yuv=check_pred_yuv, target_yuv=check_target_yuv)
    check_loss.backward()
    
    zero_grad_params = 0
    total_grad_params = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            total_grad_params += 1
            gmean = p.grad.abs().mean().item()
            if gmean < 1e-10:
                zero_grad_params += 1
                print(f"  DEAD: {name} (grad_mean={gmean:.2e})")
            else:
                print(f"  OK:   {name} (grad_mean={gmean:.2e})")
    print(f"  Summary: {total_grad_params - zero_grad_params}/{total_grad_params} params have gradients")
    if zero_grad_params > 0:
        print(f"  WARNING: {zero_grad_params} params have zero gradients!")
    else:
        print("  ALL parameters receive gradients!")
    optimizer.zero_grad()  # reset for training
    print("=" * 60 + "\n")
    
    # ── Training Loop ──
    print("Starting Training Loop...")
    print(f"  Epochs: {Config.EPOCHS} | Batch: {Config.BATCH_SIZE} | LR: {Config.LEARNING_RATE}")
    print(f"  Loss: L1 + {Config.LAMBDA_SSIM}*SSIM + 0.1*Freq")
    print(f"  Mode: FP32 (no AMP)")
    print()
    
    for epoch in range(Config.EPOCHS):
        model.train()
        
        # Update LR
        lr = get_lr_cosine_schedule(epoch, Config.EPOCHS, Config.LEARNING_RATE)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # Epoch accumulators
        epoch_loss = 0.0
        epoch_l1 = 0.0
        epoch_ssim = 0.0
        epoch_freq = 0.0
        epoch_chroma = 0.0
        epoch_psnr = 0.0
        epoch_grad_norm = 0.0
        batch_count = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        for lowlight, target in pbar:
            lowlight, target = lowlight.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass (fp32 — no AMP)
            # In training mode, model returns: rgb_enh, pred_yuv, input_yuv
            enhanced, pred_yuv, input_yuv = model(lowlight)
            
            # We don't have ground truth YUV unless we compute it here
            # Compute target YUV so we can calculate chrominance loss
            _, target_u, target_v = model.yuv_converter.rgb_to_yuv(target)
            
            # Reconstruct full target_yuv tuple for the loss function
            target_yuv = (None, target_u, target_v) # Y isn't used for chroma loss, pass None
            
            loss, loss_dict = criterion(enhanced, target, pred_yuv=pred_yuv, target_yuv=target_yuv)
            
            # Backward pass with gradient clipping
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0).item()
            optimizer.step()
            
            # Compute batch PSNR
            with torch.no_grad():
                batch_psnr = compute_psnr(
                    torch.clamp(enhanced, 0, 1), target
                )
            
            batch_count += 1
            epoch_loss += loss.item()
            epoch_l1 += loss_dict['L1']
            epoch_ssim += loss_dict['SSIM_Loss']
            epoch_freq += loss_dict['Freq_Loss']
            epoch_chroma += loss_dict.get('Chroma_Loss', 0.0)
            epoch_psnr += batch_psnr
            epoch_grad_norm += grad_norm
            
            # Show running averages
            avg_total = epoch_loss / batch_count
            avg_psnr = epoch_psnr / batch_count
            pbar.set_postfix({
                "L1": f"{epoch_l1/batch_count:.4f}",
                "Crm": f"{epoch_chroma/batch_count:.4f}",
                "PSNR": f"{avg_psnr:.1f}",
                "GNorm": f"{epoch_grad_norm/batch_count:.3f}",
                "Loss": f"{avg_total:.4f}",
                "LR": f"{lr:.6f}"
            })
        
        # ── Epoch Summary ──
        avg_loss = epoch_loss / batch_count
        avg_l1 = epoch_l1 / batch_count
        avg_ssim = epoch_ssim / batch_count
        avg_freq = epoch_freq / batch_count
        avg_chroma = epoch_chroma / batch_count
        avg_psnr = epoch_psnr / batch_count
        avg_grad = epoch_grad_norm / batch_count
        
        # Loss delta tracking
        loss_delta = 0.0
        trend_icon = ""
        if prev_loss is not None:
            loss_delta = avg_loss - prev_loss
            if loss_delta < -1e-5:
                trend_icon = "v"  # improving (down arrow)
                stagnation_count = 0
            elif loss_delta > 1e-5:
                trend_icon = "^"  # worsening (up arrow)
                stagnation_count += 1
            else:
                trend_icon = "="  # flat
                stagnation_count += 1
        prev_loss = avg_loss
        
        print(f"Epoch [{epoch+1}/{Config.EPOCHS}] "
              f"Loss: {avg_loss:.4f} {trend_icon} (delta:{loss_delta:+.5f}) | "
              f"L1: {avg_l1:.4f} | SSIM: {avg_ssim:.4f} | Freq: {avg_freq:.4f} | Crm: {avg_chroma:.4f} | "
              f"PSNR: {avg_psnr:.2f} dB | GradNorm: {avg_grad:.4f} | LR: {lr:.6f}")
        
        # Stagnation warning
        if stagnation_count >= 5:
            print(f"  WARNING: Loss has not improved for {stagnation_count} consecutive epochs!")
        if stagnation_count >= 15:
            print(f"  CRITICAL: {stagnation_count} epochs without improvement. Consider stopping.")
        
        # ── Best Model Saving ──
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.ckpt_dir, "best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'psnr': avg_psnr,
            }, best_path)
            print(f"  * New best model saved (loss: {avg_loss:.4f}, PSNR: {avg_psnr:.2f} dB)")
        
        # ── Regular Checkpoint ──
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"darkir_mod_ep_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'psnr': avg_psnr,
            }, ckpt_path)
            print(f"  Saved checkpoint to {ckpt_path}")
        
        # ── Validation Loop ──
        val_psnr_avg = 0.0
        val_ssim_avg = 0.0
        if val_loader is not None and (epoch + 1) % args.val_every == 0:
            print(f"  --- Validation (Epoch {epoch+1}) ---")
            model.eval()
            val_psnr_total = 0.0
            val_ssim_total = 0.0
            val_count = 0
            
            with torch.no_grad():
                for val_idx, val_data in enumerate(val_loader):
                    if val_idx >= 5:
                        break
                    
                    if val_dataset.has_gt:
                        val_low, val_high, val_filename = val_data
                        val_high = val_high.to(device)
                    else:
                        val_low, val_filename = val_data
                        val_high = None
                    
                    val_low = val_low.to(device)
                    enhanced = model(val_low)
                    
                    enhanced = torch.nan_to_num(enhanced, nan=0.0, posinf=1.0, neginf=0.0)
                    enhanced = torch.clamp(enhanced, 0, 1)
                    
                    if val_high is not None:
                        val_psnr_total += compute_psnr(enhanced, val_high)
                        mse_val = torch.mean((enhanced - val_high) ** 2).item()
                        val_ssim_total += 1.0 - mse_val
                        val_count += 1
                    
                    out_img = TF.to_pil_image(enhanced.squeeze(0).cpu())
                    save_name = f"ep{epoch+1}_{val_filename[0]}"
                    out_path = os.path.join(args.ckpt_dir, "val_outputs", save_name)
                    out_img.save(out_path)
            
            if val_count > 0:
                val_psnr_avg = val_psnr_total / val_count
                val_ssim_avg = val_ssim_total / val_count
                print(f"  Val PSNR: {val_psnr_avg:.2f} dB | Val quality: {val_ssim_avg:.4f}")
            
            print(f"  --- Saved to {os.path.join(args.ckpt_dir, 'val_outputs')} ---")
        
        # ── Write CSV Row ──
        csv_writer.writerow([
            epoch + 1, f"{lr:.8f}", f"{avg_loss:.6f}", f"{avg_l1:.6f}",
            f"{avg_ssim:.6f}", f"{avg_freq:.6f}", f"{avg_psnr:.4f}",
            f"{avg_grad:.6f}", f"{loss_delta:.6f}",
            f"{val_psnr_avg:.4f}" if val_psnr_avg > 0 else "",
            f"{val_ssim_avg:.6f}" if val_ssim_avg > 0 else ""
        ])
        csv_file.flush()
    
    # ── Training Complete ──
    csv_file.close()
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Training log: {csv_path}")
    print(f"  Best model: {os.path.join(args.ckpt_dir, 'best_model.pth')}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Lightweight DarkIR Mod")
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--val_every', type=int, default=5, help='Run validation logic every N epochs')
    parser.add_argument('--train_low', type=str, required=True, help='Path to training low-light images')
    parser.add_argument('--train_normal', type=str, required=True, help='Path to training normal images')
    parser.add_argument('--val_low', type=str, default='', help='Path to validation low-light images')
    parser.add_argument('--val_normal', type=str, default='', help='Path to validation normal/GT images (paired)')
    
    args = parser.parse_args()
    train(args)
