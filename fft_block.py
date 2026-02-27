import torch
import torch.nn as nn
import math

class FFTBlock(nn.Module):
    """
    Channel-wise amplitude modulation in frequency domain using rFFT2.
    Always operates in fp32 for numerical stability.
    """
    def __init__(self, channels: int, reduction_ratio: int = 4):
        super(FFTBlock, self).__init__()
        
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        
        # Channel Gating MLP 
        # C -> C // reduction_ratio -> C
        reduced_channels = max(1, channels // reduction_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.GELU(),
            nn.Linear(reduced_channels, channels, bias=True),
        )
        
        # Near-identity init: small random weights allow gradient flow,
        # bias = softplus⁻¹(1.0) ensures gating ≈ 1.0 (pass-through) at init
        inv_softplus_1 = math.log(math.e - 1.0)
        with torch.no_grad():
            nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=0.01)
            self.mlp[-1].bias.fill_(inv_softplus_1)
        self.gate_act = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: [B, C, H_d, W_d]
        Output:  [B, C, H_d, W_d]
        """
        B, C, H_d, W_d = x.shape
        orig_dtype = x.dtype
        
        # Always compute FFT in fp32 — fp16 FFT causes NaN gradients
        x_fp32 = x.float()
            
        # Step 1: Apply rFFT2
        F_spec = torch.fft.rfft2(x_fp32, norm='ortho')
        
        # Step 2: Decompose Spectrum
        amplitude = torch.abs(F_spec)
        phase = torch.angle(F_spec)
        
        # Step 3: Global Channel Energy Extraction
        energy = amplitude.mean(dim=(2, 3))  # [B, C]
        
        # Step 4: Channel Gating MLP (also in fp32)
        raw_gate = self.mlp(energy)
        gating_weights = self.gate_act(raw_gate)
        
        # Step 5: Channel-wise Scaling
        gating_weights = gating_weights.unsqueeze(-1).unsqueeze(-1)
        scaled_amplitude = amplitude * gating_weights
        
        # Step 6: Recompose Spectrum
        F_new = torch.polar(scaled_amplitude, phase)
        
        # Step 7: Inverse rFFT2
        out = torch.fft.irfft2(F_new, s=(H_d, W_d), norm='ortho')
        
        # Cast back to original dtype
        return out.to(orig_dtype)

