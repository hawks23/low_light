import torch

class Config:
    # Model parameters
    DOWNSAMPLE_FACTOR = 2   # Process FFT at half-res (was 4x, too aggressive)
    C = 48                  # Internal channels for Y branch (was 16, too narrow)
    UV_CHANNELS = 24        # Channels for UV branch (was 8)
    REDUCTION_RATIO = 4     # For FFT channel gating MLP
    FFT_BLOCKS = 1

    # Training parameters
    PATCH_SIZE = 512
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-4
    EPOCHS = 300
    BATCH_SIZE = 8          # Doubled for more stable gradients
    # Loss weights
    LAMBDA_SSIM = 0.1       # Low initially — SSIM is noisy for dark images
    LAMBDA_FREQ = 1.0       # Increased to 1.0 to strongly penalize blurriness and encourage sharp edges
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @classmethod
    def strict_efficiency(cls):
        """Reduces channel counts for stricter efficiency as per coding plan."""
        cls.C = 12
        cls.UV_CHANNELS = 6
