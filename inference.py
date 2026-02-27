import os
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import argparse
import time
from torcheval.metrics.functional.image import peak_signal_noise_ratio as _psnr
from ptflops import get_model_complexity_info # Requires `pip install ptflops` for MACs/FLOPs profiling

from config import Config
from model import LightweightDarkIR
from losses import SSIMLoss # Repurposing the SSIM loss as a metric

def evaluate(args):
    device = Config.DEVICE
    print(f"Using device: {device}")
    
    # Init Model
    model = LightweightDarkIR(
        c=Config.C, 
        downsample_factor=Config.DOWNSAMPLE_FACTOR,
        uv_channels=Config.UV_CHANNELS,
        reduction_ratio=Config.REDUCTION_RATIO
    ).to(device)
    
    # Load Checkpoint
    if args.ckpt_path and os.path.exists(args.ckpt_path):
        checkpoint = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {args.ckpt_path}")
    else:
        print("WARNING: No valid checkpoint provided. Running with untrained weights for profiling.")
        
    model.eval()
    
    # Metrics setup
    ssim_metric = SSIMLoss(size_average=True).to(device)
    
    psnr_vals = []
    ssim_vals = []
    runtimes = []
    
    # Dummy creation of test tensor if no image provided (for profiling)
    with torch.no_grad():
        if args.input:
            print(f"Loading input image: {args.input}")
            img_in = Image.open(args.input).convert('RGB')
            tensor_in = TF.to_tensor(img_in).unsqueeze(0).to(device)
            
            # Optionally load target for metrics
            if args.target:
                img_gt = Image.open(args.target).convert('RGB')
                tensor_gt = TF.to_tensor(img_gt).unsqueeze(0).to(device)
            else:
                tensor_gt = None
                
            # Warmup
            _ = model(tensor_in)
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            torch.cuda.synchronize()
            start_event.record()
            
            # Using AMP for speed if requested
            with torch.cuda.amp.autocast(enabled=args.fp16):
                out = model(tensor_in)
                
            end_event.record()
            torch.cuda.synchronize()
            
            runtime = start_event.elapsed_time(end_event) # In milliseconds
            runtimes.append(runtime)
            print(f"Inference Time: {runtime:.2f} ms")
            
            out = torch.clamp(out, 0, 1)
            
            if tensor_gt is not None:
                psnr = _psnr(out, tensor_gt).item()
                # SSIMLoss returns 1 - ssim
                ssim = 1.0 - ssim_metric(out, tensor_gt).item()
                
                print(f"Metrics -> PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
            
            if args.save_path:
                out_img = TF.to_pil_image(out.squeeze(0).cpu())
                out_img.save(args.save_path)
                print(f"Saved enhanced image to {args.save_path}")
                
        else:
            print("No input specified. Running 4K Inference Profiling Dummy Test.")
            tensor_in = torch.rand(1, 3, 2160, 3840).to(device)
            tensor_gt = torch.rand(1, 3, 2160, 3840).to(device)
            
            # Warmup
            print("Warming up GPU...")
            for _ in range(3):
                _ = model(tensor_in)
                
            runs = 10
            print(f"Running {runs} iterations of 4K forward pass...")
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            for _ in range(runs):
                torch.cuda.synchronize()
                start_event.record()
                
                with torch.cuda.amp.autocast(enabled=args.fp16):
                    out = model(tensor_in)
                    
                end_event.record()
                torch.cuda.synchronize()
                
                runtimes.append(start_event.elapsed_time(end_event))
            
            avg_runtime = sum(runtimes) / len(runtimes)
            print(f"Average 4K Inference Time: {avg_runtime:.2f} ms | FPS: {1000/avg_runtime:.2f}")
            
            # Dummy PSNR/SSIM calculation to show they work
            out = torch.clamp(out, 0, 1)
            psnr = _psnr(out, tensor_gt).item()
            ssim = 1.0 - ssim_metric(out, tensor_gt).item()
            print(f"Dummy Metrics -> PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
            
    # FLOPs profiling
    if args.metrics:
        print("\nProfiling FLOPs and Memory...")
        try:
            # ptflops requires the module to be initialized and input dimensions provided
            macs, params = get_model_complexity_info(
                model, (3, 2160, 3840), 
                as_strings=True, print_per_layer_stat=False, verbose=False
            )
            print(f"Computational complexity: {macs}")
            print(f"Number of parameters: {params}")
        except ImportError:
            print("Error: `ptflops` library not found. Install via `pip install ptflops` for MACs profiling.")
        
        # Max GPU Mem
        if torch.cuda.is_available():
            max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"Max GPU Memory Allocated: {max_mem:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Lightweight DarkIR Mod")
    parser.add_argument('--input', type=str, help='Path to input image')
    parser.add_argument('--target', type=str, help='Path to ground truth image for metrics')
    parser.add_argument('--save_path', type=str, help='Path to save output image')
    parser.add_argument('--ckpt_path', type=str, help='Path to checkpoint file')
    parser.add_argument('--metrics', action='store_true', help='Calculate and display FLOPs and Memory complexity')
    parser.add_argument('--fp16', action='store_true', help='Use Mixed Precision (FP16) inference')
    
    args = parser.parse_args()
    evaluate(args)
