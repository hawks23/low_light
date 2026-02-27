import torch
import torch.nn as nn
from yuv_conversion import YUVConversion
from y_branch import YBranch
from uv_branch import UVBranch
from config import Config


class LightweightDarkIR(nn.Module):
    """
    YUV-Luminance FFT Low-Light Enhancement Model.
    
    Architecture: RGB → YUV → Y-branch + UV-branch → YUV → RGB correction
    Output = original_input + learned_correction (global residual)
    """
    def __init__(self, c: int = Config.C, 
                 downsample_factor: int = Config.DOWNSAMPLE_FACTOR,
                 uv_channels: int = Config.UV_CHANNELS, 
                 reduction_ratio: int = Config.REDUCTION_RATIO):
        super(LightweightDarkIR, self).__init__()
        
        # Color space conversion (stateless, deterministic)
        self.yuv_converter = YUVConversion()
        
        # Y Branch: luminance enhancement via FFT amplitude gating
        self.y_branch = YBranch(
            channels=c, 
            downsample_factor=downsample_factor, 
            reduction_ratio=reduction_ratio
        )
        
        # UV Branch: chroma stabilization guided by enhanced Y
        self.uv_branch = UVBranch(uv_channels=uv_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, H, W] in [0, 1]
        Returns: 
        - rgb_enh: Enhanced RGB image [B, 3, H, W]
        - pred_yuv: Tuple of (Y, U, V) enhanced tensors
        - input_yuv: Tuple of (Y, U, V) original tensors
        """
        # 1. RGB -> YUV decomposition
        y, u, v = self.yuv_converter.rgb_to_yuv(x)
        
        # 2. Luminance enhancement (learns brightness correction in frequency domain)
        #    Returns enhanced Y and the rich spatial feature map for the UV branch
        y_enh, y_features = self.y_branch(y)
        
        # 3. Chroma correction (guided by Y branch features)
        u_enh, v_enh = self.uv_branch(u, v, y_features)
        
        # 4. Convert enhanced YUV back to RGB
        #    The Y and UV branches already have residual connections,
        #    so rgb_enh ≈ x + learned_correction (no separate refine needed)
        rgb_enh = self.yuv_converter.yuv_to_rgb(y_enh, u_enh, v_enh)
        
        if self.training:
            return rgb_enh, (y_enh, u_enh, v_enh), (y, u, v)
        return rgb_enh


if __name__ == "__main__":
    import time
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightweightDarkIR().to(device)
    model.eval()
    
    # Parameter count
    params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {params:,}")
    print(f"Under 80K: {'PASS' if params < 80000 else 'FAIL'}")
    
    # Quick shape test
    test = torch.rand(1, 3, 512, 512).to(device)
    with torch.no_grad():
        start = time.time()
        out = model(test)
        elapsed = (time.time() - start) * 1000
        print(f"512x512 Output Shape: {out.shape}")
        print(f"512x512 Forward Time: {elapsed:.2f} ms")
        
        # Verify output is close to input at initialization (global residual property)
        diff = (out - test).abs().mean().item()
        print(f"Mean diff from input (should be small at init): {diff:.6f}")
