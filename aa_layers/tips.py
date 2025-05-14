import torch
import torch.nn as nn
import torch.nn.functional as F

class TIPS2d(nn.Module):
    """
    Translation Invariant Polyphase Sampling (TIPS) 2D layer
    
    A downsampling layer that provides improved shift equivariance.
    Based on the paper "TIPS: Translation Invariant Polyphase Sampling for Dense Prediction Tasks"
    
    Args:
        kernel_size: Kernel size for pooling
        stride: Stride for pooling
        padding: Padding for pooling
        sigma: Gaussian blur parameter, higher values give more blurring
    """
    def __init__(self, kernel_size=2, stride=2, padding=0, sigma=0.8):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.sigma = sigma
        
        # If kernel_size is 1, TIPS is equivalent to identity
        if kernel_size == 1 or stride == 1:
            self.tips_kernel = None
        else:
            # Create TIPS kernel
            self.tips_kernel = self._create_tips_kernel()
    
    def _create_tips_kernel(self):
        """Create the TIPS sampling kernel based on Gaussian filter"""
        kernel_size = self.kernel_size
        sigma = self.sigma
        
        # Create Gaussian kernel
        grid_x = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
        grid_y = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
        
        x, y = torch.meshgrid(grid_x, grid_y, indexing='ij')
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()  # Normalize
        
        # Reshape to 4D conv kernel format [out_channels, in_channels, H, W]
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        
        return nn.Parameter(kernel, requires_grad=False)
    
    def forward(self, x):
        if self.tips_kernel is None:
            # No downsampling - just pass through
            return x
        
        # Apply Gaussian blur
        x = F.conv2d(x, self.tips_kernel.expand(x.size(1), 1, -1, -1).transpose(0, 1),
                     padding=self.padding, stride=self.stride, groups=x.size(1))
        
        return x 