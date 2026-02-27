import torch
import torch.nn as nn

class UVBranch(nn.Module):
    """
    UV Branch: Lightweight Y-guided chroma stabilization.
    """
    def __init__(self, uv_channels: int = 8, y_feat_channels: int = 48):
        super(UVBranch, self).__init__()
        
        # We project the Y features down to a small number of channels 
        # to keep the overall parameter count well under 80K
        projected_y_channels = 8
        self.y_proj = nn.Conv2d(y_feat_channels, projected_y_channels, kernel_size=1, bias=False)
        
        # Input channels: U(1) + V(1) + projected_Y_features(8) = 10
        # Output channels: delta_U(1) + delta_V(1) = 2
        in_channels = 2 + projected_y_channels
        
        # Global context path to fix image-wide color casts (< 200 params)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_context = nn.Sequential(
            nn.Conv2d(in_channels, uv_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(uv_channels, in_channels, kernel_size=1, bias=True)
        )
        # Init color shift to near-zero so it doesn't destabilize early training,
        # but using small random weights to keep the gradient path alive.
        with torch.no_grad():
            nn.init.normal_(self.global_context[-1].weight, mean=0.0, std=0.001)
            nn.init.zeros_(self.global_context[-1].bias)
        
        self.net = nn.Sequential(
            # Standard Conv 3x3 (in_channels -> uv_channels)
            nn.Conv2d(in_channels, uv_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            
            # Standard Conv 3x3 (uv_channels -> uv_channels)
            nn.Conv2d(uv_channels, uv_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            
            # Final Conv 1x1 (uv_channels -> 2)
            nn.Conv2d(uv_channels, 2, kernel_size=1, stride=1, padding=0, bias=True)
        )
        
        # Near-identity init: small random weights allow gradient flow,
        # bias stays zero so delta_u ≈ 0, delta_v ≈ 0 at init
        with torch.no_grad():
            nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.001)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, u: torch.Tensor, v: torch.Tensor, y_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs: 
        - u, v: Original U, V channels [B, 1, H, W]
        - y_features: Spatial features from Y branch [B, C_y, H, W]
        Returns:
        - u_enh, v_enh: Enhanced U and V channels [B, 1, H, W]
        """
        # Project Y features to save parameters
        y_proj = self.y_proj(y_features)  # [B, 8, H, W]
        
        # Concatenate inputs along channel dimension: [1] + [1] + [8] = [10]
        features = torch.cat([u, v, y_proj], dim=1)  # [B, 10, H, W]
        
        # Extract global color prior and broadcast-add it back as a spatial shift
        global_feat = self.global_pool(features)                 # [B, 10, 1, 1]
        color_shift = self.global_context(global_feat)           # [B, 10, 1, 1]
        features = features + color_shift                        # [B, 10, H, W]
        
        # Forward pass
        deltas = self.net(features)  # [B, 2, H, W]
        
        delta_u, delta_v = torch.split(deltas, 1, dim=1)
        
        # Residual reconstruction
        u_enh = u + delta_u
        v_enh = v + delta_v
        
        return u_enh, v_enh
