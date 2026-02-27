import torch
import torch.nn as nn
import torch.nn.functional as F

class SSIMLoss(nn.Module):
    """
    Simplified SSIM loss for training. 
    Allows for differential calculation across patches.
    """
    def __init__(self, window_size: int = 11, size_average: bool = True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([torch.exp(torch.tensor(-(x - window_size//2)**2/float(2*sigma**2))) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        # Return 1 - SSIM to act as a loss to minimize
        return 1.0 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

class DarkIRLoss(nn.Module):
    """
    Combined Loss for Lightweight DarkIR.
    L = L1 + lambda_ssim * SSIM + (lambda_freq * FreqAbsLoss)
    """
    def __init__(self, lambda_ssim: float = 0.1, lambda_freq: float = 0.1):
        super(DarkIRLoss, self).__init__()
        self.lambda_ssim = lambda_ssim
        self.lambda_freq = lambda_freq
        
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                pred_yuv: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
                target_yuv: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None) -> tuple[torch.Tensor, dict]:
        """
        Returns total loss and a dictionary of individual components.
        Optionally takes YUV components to compute chrominance loss.
        """
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        
        # Optional: Frequency Domain Consistency Term
        # Encourage the predicted spectrum to match the target spectrum in the luminance channel
        # Use float32 explicitly to prevent FP16 overflow resulting in NaNs during FFT magnitude extraction
        with torch.amp.autocast('cuda', enabled=False):
            pred_fp32 = pred.float()
            target_fp32 = target.float()
            
            pred_fft = torch.fft.rfft2(pred_fp32, norm='ortho')
            target_fft = torch.fft.rfft2(target_fp32, norm='ortho')
            
            # Add eps to prevent inf/nan gradients for zero amplitude 
            pred_amp = torch.abs(pred_fft) + 1e-8
            target_amp = torch.abs(target_fft) + 1e-8
            
        freq_loss = self.l1_loss(pred_amp, target_amp)
        
        # Chrominance Loss (if YUV components are provided)
        chroma_loss = torch.tensor(0.0, device=pred.device)
        if pred_yuv is not None and target_yuv is not None:
            _, p_u, p_v = pred_yuv
            _, t_u, t_v = target_yuv
            # Weight UV errors slightly higher since their scale is smaller than Y
            chroma_loss = (self.l1_loss(p_u, t_u) + self.l1_loss(p_v, t_v)) * 0.5
        
        # Guard against NaNs in SSIM gracefully
        if torch.isnan(ssim) or torch.isinf(ssim):
            ssim = torch.tensor(0.0, device=pred.device)
            
        total_loss = l1 + (self.lambda_ssim * ssim) + (self.lambda_freq * freq_loss) + (0.5 * chroma_loss)
        
        loss_dict = {
            "L1": l1.item(),
            "SSIM_Loss": ssim.item(),
            "Freq_Loss": freq_loss.item(),
            "Chroma_Loss": chroma_loss.item(),
            "Total": total_loss.item()
        }
        
        return total_loss, loss_dict
