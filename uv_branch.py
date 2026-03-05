import torch
import torch.nn as nn

class UVBranch(nn.Module):
    """
    UV Branch: Spatial Feature Transform (SFT) Modulation.
    Dynamically predicts spatial scale (gamma) and shift (beta) matrices 
    for the UV channels directly from the enhanced luminance map (Y).
    """
    def __init__(self, uv_channels: int = 16, y_feat_channels: int = 48):
        super(UVBranch, self).__init__()
        
        # 1. Base Feature Projection 
        # Compress the rich Y-features from the Y-branch down to 8 channels
        projected_y_channels = 8
        self.y_proj = nn.Conv2d(y_feat_channels, projected_y_channels, kernel_size=1, bias=False)
        
        # Input channels into the SFT generator:
        # projected_Y_features(8) + Y_enh(1) = 9 channels
        sft_in_channels = projected_y_channels + 1
        
        # 2. SFT Parameter Generator (< 5,000 params)
        # Uses Dilated Depthwise Convolutions to expand the receptive field to 5x5
        # without adding parameters or blurring the outputs.
        # This network "looks" at the lighting (Y_enh) and the geometry (Y_features)
        # to decide where to amplify color (gamma) and where to shift it (beta).
        self.sft_net = nn.Sequential(
            # Expand to uv_channels
            nn.Conv2d(sft_in_channels, uv_channels, kernel_size=1, bias=False),
            
            # Dilated Depthwise (spatial analysis of lighting)
            nn.Conv2d(uv_channels, uv_channels, kernel_size=3, padding=2, dilation=2, groups=uv_channels, bias=False),
            nn.GELU(),
            
            # Compress to exactly 4 channels: gamma_u, gamma_v, beta_u, beta_v
            nn.Conv2d(uv_channels, 4, kernel_size=1, bias=True)
        )

        with torch.no_grad():
            # Init SFT generator near zero so the network starts as an Identity function:
            # gamma ≈ 0, beta ≈ 0  => UV_out = UV_in * (1 + 0) + 0 = UV_in
            nn.init.normal_(self.sft_net[-1].weight, mean=0.0, std=0.001)
            nn.init.zeros_(self.sft_net[-1].bias)

    def forward(self, u: torch.Tensor, v: torch.Tensor, y_features: torch.Tensor, y_enh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs: 
        - u, v: Original U, V channels [B, 1, H, W]
        - y_features: Spatial features from Y branch [B, C_y, H, W]
        - y_enh: The fully completed, enhanced luminance map [B, 1, H, W]
        """
        # --- Phase 1: Context Gathering ---
        y_proj = self.y_proj(y_features)                # [B, 8, H, W]
        
        # The SFT Generator only looks at Y-data (geometry + lighting) to make decisions
        sft_context = torch.cat([y_proj, y_enh], dim=1) # [B, 9, H, W]
        
        # --- Phase 2: SFT Parameter Prediction ---
        # Generate the spatial modulation grids
        sft_params = self.sft_net(sft_context)          # [B, 4, H, W]
        
        # Split into scale (gamma) and shift (beta) for U and V
        gamma_u = sft_params[:, 0:1, :, :]
        gamma_v = sft_params[:, 1:2, :, :]
        beta_u  = sft_params[:, 2:3, :, :]
        beta_v  = sft_params[:, 3:4, :, :]
        
        # --- Phase 3: Spatial Feature Transform (SFT) Modulation ---
        # UV_out = UV_in * (1 + gamma) + beta
        u_final = u * (1.0 + gamma_u) + beta_u
        v_final = v * (1.0 + gamma_v) + beta_v
        
        return u_final, v_final
