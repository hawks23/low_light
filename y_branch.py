import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from fft_block import FFTBlock


class InvertedResidual(nn.Module):
    """1×1 expand → DW 3×3 + GELU → 1×1 compress + residual skip.

    Spatial ops are depthwise (no cross-channel spatial mixing).
    Channel adaptation through 1×1 pointwise only.
    """
    def __init__(self, channels: int, expand_ratio: int = 4):
        super().__init__()
        mid = channels * expand_ratio
        self.block = nn.Sequential(
            # Pointwise expand: C → C*expand
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.GELU(),
            # Depthwise spatial: C*expand → C*expand (per-channel, no mixing)
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),
            nn.GELU(),
            # Pointwise compress: C*expand → C
            nn.Conv2d(mid, channels, 1, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)  # residual skip

class YBranch(nn.Module):
    """
    Y Branch: Hybrid illumination correction (multiplicative + additive).
    
    Multiplicative: handles already-visible pixels (scale proportional to Y)
    Additive: handles near-zero pixels where multiplication can't inject light
    
    y_enh = y * softplus(scale_raw) + sigmoid(offset_raw) * max_offset
    """
    def __init__(self, channels: int = 16, downsample_factor: int = 4, 
                 reduction_ratio: int = 4, max_offset: float = 0.5):
        super(YBranch, self).__init__()
        
        self.channels = channels
        self.downsample_factor = downsample_factor
        self.max_offset = max_offset
        
        # Step 2: Feature Projection (1 -> C)
        self.proj_in = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True)
        )
        
        # Step 3: Inverted Residual blocks (MobileNetV2 pattern)
        expand = 4
        self.spatial_block = nn.Sequential(
            InvertedResidual(channels, expand),
            InvertedResidual(channels, expand),
        )
        
        # Step 4: FFT Block
        self.fft_block = FFTBlock(channels=channels, reduction_ratio=reduction_ratio)
        
        # Step 5: Reprojection (C -> 2: scale_raw + offset_raw)
        self.proj_out = nn.Conv2d(channels, 2, kernel_size=1, stride=1, padding=0, bias=True)
        
        # Anti-aliasing post-upsample filter to prevent moiré artifacts (~18 params)
        self.aa_conv = nn.Conv2d(2, 2, kernel_size=3, padding=1, groups=2, bias=False)
        
        # Near-identity init: small random weights allow gradient flow,
        # bias controls initial behavior:
        #   scale channel: softplus(bias) ≈ 1.0 → y * 1.0 = y
        #   offset channel: sigmoid(bias) ≈ 0.0 → additive ≈ 0
        inv_softplus_1 = math.log(math.e - 1.0)  # ≈ 0.5413
        with torch.no_grad():
            nn.init.normal_(self.proj_out.weight, mean=0.0, std=0.01)
            # Scale channel bias → softplus(0.5413) ≈ 1.0
            self.proj_out.bias[0] = inv_softplus_1
            # Offset channel bias → sigmoid(-5) ≈ 0.007 → near-zero offset
            self.proj_out.bias[1] = -5.0
            
            # Init aa_conv to identity (center pixel = 1) to not disrupt starting state
            self.aa_conv.weight.zero_()
            for i in range(2):
                self.aa_conv.weight[i, 0, 1, 1] = 1.0

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Input y: [B, 1, H, W]
        Returns: 
        - Y_enh: [B, 1, H, W] (Enhanced Y channel)
        - scale_map: [B, 1, H, W] (Scale map for diagnostics)
        """
        # Step 1: Downsampling
        _, _, H, W = y.shape
        H_d = H // self.downsample_factor
        W_d = W // self.downsample_factor
        
        # Interpolate with area sampling to naturally avoid aliasing (Moiré) during downsample
        y_down = F.interpolate(y, size=(H_d, W_d), mode='area')
        
        # Step 2: Feature Projection
        feat = self.proj_in(y_down)  # [B, C, H_d, W_d]
        
        # Step 3: Multi-layer spatial feature extraction
        feat = self.spatial_block(feat)  # [B, C, H_d, W_d]
        
        # Step 4: FFT Block (with residual skip for stability)
        feat = feat + self.fft_block(feat)  # [B, C, H_d, W_d]
        
        # Step 5: Reprojection to 2-channel correction signal
        corrections_d = self.proj_out(feat)  # [B, 2, H_d, W_d]
        
        # Step 6: Upsampling and Anti-Aliasing
        corrections = F.interpolate(corrections_d, size=(H, W), mode='bilinear', align_corners=False)
        corrections = self.aa_conv(corrections)
        
        # Splitting corrections...
        scale_raw = corrections[:, 0:1, :, :]   # [B, 1, H, W]
        offset_raw = corrections[:, 1:2, :, :]  # [B, 1, H, W]
        
        # Step 7: HYBRID enhancement
        scale_map = F.softplus(scale_raw)
        offset_map = torch.sigmoid(offset_raw) * self.max_offset
        
        y_enh = y * scale_map + offset_map
        
        # Upsample the C-channel feature map to match H, W so UV branch can use it
        feat_up = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)

        return y_enh, feat_up
