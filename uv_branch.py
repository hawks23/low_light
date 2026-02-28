import torch
import torch.nn as nn

class UVBranch(nn.Module):
    """
    UV Branch: Y-Guided High-Frequency Re-injection.
    Uses dilated depthwise convolutions for artifact smoothing (moiré/checkerboard)
    and pristine Y-edges to explicitly reinject high-frequency color without bleeding.
    """
    def __init__(self, uv_channels: int = 16, y_feat_channels: int = 48):
        super(UVBranch, self).__init__()
        
        # 1. Project rich Y-features down to extract geometric guidance
        projected_y_channels = 8
        self.y_proj = nn.Conv2d(y_feat_channels, projected_y_channels, kernel_size=1, bias=False)
        
        # 2. Global Color Cast Fix (< 200 params)
        # Input: U(1) + V(1) + Y_proj(8) = 10
        in_channels = 2 + projected_y_channels
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_context = nn.Sequential(
            nn.Conv2d(in_channels, uv_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(uv_channels, in_channels, kernel_size=1, bias=True)
        )
        
        # 3. Artifact Smoothing (Moiré & Checkerboard destruction)
        # Uses Dilated Depthwise Convs to get a 5x5 receptive field super cheaply.
        self.smoother = nn.Sequential(
            # Expand to uv_channels
            nn.Conv2d(in_channels, uv_channels, kernel_size=1, bias=False),
            # Dilated Depthwise (spatially smooths U and V independently using Y geometry)
            nn.Conv2d(uv_channels, uv_channels, kernel_size=3, padding=2, dilation=2, groups=uv_channels, bias=False),
            nn.GELU(),
            # Compress to 2 channels (smoothed delta_u, delta_v)
            nn.Conv2d(uv_channels, 2, kernel_size=1, bias=True)
        )
        
        # 4. High-Frequency Edge Extract & Inject (SpADE/AdaIN style)
        # Takes the perfect high-freq Y-edges and generates spatial multipliers for original U/V
        self.hf_injector = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(8, 2, kernel_size=1, bias=True)
        )

        with torch.no_grad():
            # Init global bias near zero
            nn.init.normal_(self.global_context[-1].weight, mean=0.0, std=0.001)
            nn.init.zeros_(self.global_context[-1].bias)
            
            # Init smoother near zero to act as identity
            nn.init.normal_(self.smoother[-1].weight, mean=0.0, std=0.001)
            nn.init.zeros_(self.smoother[-1].bias)
            
            # Init HF injector near zero
            nn.init.normal_(self.hf_injector[-1].weight, mean=0.0, std=0.001)
            nn.init.zeros_(self.hf_injector[-1].bias)

    def extract_high_frequencies(self, y_enh: torch.Tensor) -> torch.Tensor:
        """Deterministic, zero-param Laplacian-style high frequency extraction."""
        # Simple cross-shaped Laplacian kernel to find sharp edges
        kernel = torch.tensor([[[[0.0, -1.0, 0.0],
                                 [-1.0, 4.0, -1.0],
                                 [0.0, -1.0, 0.0]]]], device=y_enh.device, dtype=y_enh.dtype)
        # Pad to keep spatial dimensions identical
        y_hf = torch.nn.functional.conv2d(y_enh, kernel, padding=1)
        return y_hf

    def forward(self, u: torch.Tensor, v: torch.Tensor, y_features: torch.Tensor, y_enh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs: 
        - u, v: Original U, V channels [B, 1, H, W]
        - y_features: Spatial features from Y branch [B, C_y, H, W]
        - y_enh: The fully completed, enhanced luminance map [B, 1, H, W]
        """
        # --- Phase 1: Guided Global Color Cast Fix ---
        y_proj = self.y_proj(y_features)              # [B, 8, H, W]
        features = torch.cat([u, v, y_proj], dim=1)   # [B, 10, H, W]
        
        global_feat = self.global_pool(features)                 
        color_shift = self.global_context(global_feat)           
        features = features + color_shift                        
        
        # --- Phase 2: Artifact Smoothing ---
        # Dilated convolutions smear out moiré artifacts and checkerboards
        deltas = self.smoother(features)              # [B, 2, H, W]
        delta_u, delta_v = torch.split(deltas, 1, dim=1)
        
        u_smooth = u + delta_u
        v_smooth = v + delta_v
        
        # --- Phase 3 & 4: HF Edge Extraction & Re-injection ---
        y_hf = self.extract_high_frequencies(y_enh)   # [B, 1, H, W] - Pristine sharp edges
        
        # Generate explicit spatial attention maps for high-frequency details
        hf_gamma = self.hf_injector(y_hf)             # [B, 2, H, W]
        gamma_u, gamma_v = torch.split(hf_gamma, 1, dim=1)
        
        # Multiply original high-res color by the edge weights, and add to the smoothed base
        u_final = u_smooth + (u * gamma_u)
        v_final = v_smooth + (v * gamma_v)
        
        return u_final, v_final
