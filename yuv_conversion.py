import torch
import torch.nn as nn

class YUVConversion:
    """
    Deterministic RGB to YUV and YUV to RGB conversion based on BT.601 standard.
    """
    def __init__(self):
        # RGB to YUV conversion matrix (BT.601 standard)
        self.rgb_to_yuv_mat = torch.tensor([
            [0.299, 0.587, 0.114],
            [-0.14713, -0.28886, 0.436],
            [0.615, -0.51499, -0.10001]
        ], dtype=torch.float32)

        # YUV to RGB conversion matrix
        self.yuv_to_rgb_mat = torch.tensor([
            [1.0, 0.0, 1.13983],
            [1.0, -0.39465, -0.58060],
            [1.0, 2.03211, 0.0]
        ], dtype=torch.float32)

    def rgb_to_yuv(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Converts RGB [B, 3, H, W] to Y, U, V each [B, 1, H, W]
        Assumes RGB is in range [0, 1]
        """
        B, C, H, W = rgb.shape
        assert C == 3, "Input must have 3 channels (RGB)"
        
        # Move matrix to the same device and dtype as input
        device_mat = self.rgb_to_yuv_mat.to(rgb.device, dtype=rgb.dtype)
        
        # Flatten spatial dimensions for matrix multiplication
        rgb_flat = rgb.view(B, 3, -1)  # [B, 3, H*W]
        
        # Apply transformation: YUV = Mat * RGB
        # Mat: [3, 3], rgb_flat: [B, 3, H*W] -> [B, 3, H*W]
        yuv_flat = torch.matmul(device_mat, rgb_flat)
        
        # Reshape back to spatial dimensions
        yuv = yuv_flat.view(B, 3, H, W)
        
        # Split into Y, U, V components [B, 1, H, W] each
        y, u, v = torch.split(yuv, 1, dim=1)
        
        return y, u, v

    def yuv_to_rgb(self, y: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Converts Y, U, V [B, 1, H, W] back to RGB [B, 3, H, W]
        """
        # Concatenate back to [B, 3, H, W]
        yuv = torch.cat([y, u, v], dim=1)
        
        B, C, H, W = yuv.shape
        assert C == 3, "Input must have 3 channels (YUV)"
        
        # Move matrix to the same device and dtype as input
        device_mat = self.yuv_to_rgb_mat.to(yuv.device, dtype=yuv.dtype)
        
        # Flatten spatial dimensions
        yuv_flat = yuv.view(B, 3, -1)  # [B, 3, H*W]
        
        # Apply transformation: RGB = Mat * YUV
        rgb_flat = torch.matmul(device_mat, yuv_flat)
        
        # Reshape back
        rgb = rgb_flat.view(B, 3, H, W)
        
        # Ensure values are within valid ranges [0,1] due to minor precision issues
        # But we won't strictly clamp during intermediate processing to allow gradients to flow freely
        # Only clamp right before output or saving if needed.
        
        return rgb

# Quick test script
if __name__ == "__main__":
    converter = YUVConversion()
    
    # Create random standard float32 tensor
    rgb = torch.rand(2, 3, 256, 256)
    
    # Forward pass
    y, u, v = converter.rgb_to_yuv(rgb)
    print(f"Y shape: {y.shape}, U shape: {u.shape}, V shape: {v.shape}")
    
    # Backward pass
    rgb_recon = converter.yuv_to_rgb(y, u, v)
    print(f"RGB Recon shape: {rgb_recon.shape}")
    
    # Assert numerical stability
    diff = torch.abs(rgb - rgb_recon).max().item()
    print(f"Max reconstruction error: {diff:.6f}")
    assert diff < 1e-5, "Reconstruction error too high"

    # Test float16
    rgb_fp16 = rgb.half().cuda() if torch.cuda.is_available() else rgb.half()
    if torch.cuda.is_available(): # Only run fp16 test if CUDA is available, CPU half is slow/sometimes unsupported for matmul
        y_h, u_h, v_h = converter.rgb_to_yuv(rgb_fp16)
        recon_fp16 = converter.yuv_to_rgb(y_h, u_h, v_h)
        diff_fp16 = torch.abs(rgb_fp16 - recon_fp16).max().item()
        print(f"FP16 max error: {diff_fp16:.6f}")
