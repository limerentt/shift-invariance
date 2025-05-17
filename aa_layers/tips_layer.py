#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implementation of Translation Invariant Polyphase Sampling (TIPS) layers.
Based on "TIPS: Translation Invariant Polyphase Sampling" (Saha & Gokhale, 2024).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Union, Tuple, Optional

class TIPSConv2d(nn.Module):
    """
    Translation Invariant Polyphase Sampling Convolution.
    
    This layer implements TIPS convolution which produces shift-invariant features
    by learning phase-specific weights for all possible sub-pixel offsets.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of convolution kernel
        stride (int): Stride for downsampling (default: 1)
        padding (int): Padding (default: 0)
        dilation (int): Dilation factor (default: 1)
        groups (int): Number of groups (default: 1)
        bias (bool): Whether to include bias (default: True)
        padding_mode (str): Padding mode ('zeros', 'reflect', 'replicate', 'circular') (default: 'zeros')
        num_phases (int): Number of phase filters to use (default: None, which sets to stride^2)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = True, padding_mode: str = 'zeros',
                 num_phases: Optional[int] = None):
        super(TIPSConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        
        # Compute number of phases
        self.phases_h = self.stride[0]
        self.phases_w = self.stride[1]
        self.num_phases = num_phases if num_phases is not None else (self.phases_h * self.phases_w)
        
        # Create a standard convolution for each phase
        self.phase_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, self.kernel_size,
                     stride=1, padding=self.padding, dilation=self.dilation,
                     groups=self.groups, bias=bias, padding_mode=padding_mode)
            for _ in range(self.num_phases)
        ])
        
        # Learnable phase weights (initialized equally)
        self.phase_weights = nn.Parameter(torch.ones(self.num_phases) / self.num_phases)
        
        # Downsampling function (if stride > 1)
        self.need_downsample = stride > 1
        if self.need_downsample:
            self.downsample = TIPSDownsampler(out_channels, self.stride)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Output tensor
        """
        batch_size, _, h, w = x.shape
        
        if self.stride[0] == 1 and self.stride[1] == 1:
            # For stride=1, just apply a single convolution (no phases needed)
            return self.phase_convs[0](x)
        
        # Compute phase-specific features
        all_phases = []
        phase_idx = 0
        
        for i in range(self.phases_h):
            for j in range(self.phases_w):
                if phase_idx >= self.num_phases:
                    break
                    
                # Apply phase-specific shift (sub-pixel)
                offset_h = i / self.stride[0]
                offset_w = j / self.stride[1]
                
                # Fractional shift using interpolation
                shifted_x = self._fractional_shift(x, offset_h, offset_w)
                
                # Apply phase convolution
                phase_output = self.phase_convs[phase_idx](shifted_x)
                
                # Apply phase weight
                phase_output = phase_output * self.phase_weights[phase_idx]
                
                all_phases.append(phase_output)
                phase_idx += 1
        
        # Sum all phase outputs
        y = torch.stack(all_phases).sum(dim=0)
        
        # Apply downsampling if needed
        if self.need_downsample:
            y = self.downsample(y)
        
        return y
    
    def _fractional_shift(self, x, offset_h, offset_w):
        """
        Apply sub-pixel shift to input tensor using bilinear interpolation.
        
        Args:
            x (torch.Tensor): Input tensor
            offset_h (float): Vertical shift (fraction of pixel)
            offset_w (float): Horizontal shift (fraction of pixel)
            
        Returns:
            torch.Tensor: Shifted tensor
        """
        batch_size, channels, h, w = x.shape
        
        # Create sampling grid
        grid_h, grid_w = torch.meshgrid(
            torch.arange(h, device=x.device, dtype=torch.float32),
            torch.arange(w, device=x.device, dtype=torch.float32)
        )
        
        # Apply offset
        grid_h = grid_h - offset_h
        grid_w = grid_w - offset_w
        
        # Normalize grid coordinates to [-1, 1]
        grid_h = 2 * grid_h / (h - 1) - 1
        grid_w = 2 * grid_w / (w - 1) - 1
        
        # Combine into grid
        grid = torch.stack([grid_w, grid_h], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        # Apply grid sampling
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)


class TIPSDownsampler(nn.Module):
    """
    Translation Invariant Polyphase Sampling Downsampler.
    
    This module implements shift-invariant downsampling using polyphase filters.
    
    Args:
        channels (int): Number of input channels
        stride (Union[int, Tuple[int, int]]): Downsampling factor (default: 2)
        kernel_size (int): Size of the anti-aliasing kernel (default: 3)
    """
    def __init__(self, channels: int, stride: Union[int, Tuple[int, int]] = 2, kernel_size: int = 3):
        super(TIPSDownsampler, self).__init__()
        
        self.channels = channels
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.kernel_size = kernel_size
        
        # Create sinc kernel
        k = self._create_sinc_kernel(kernel_size, self.stride[0])
        self.register_buffer('kernel', k.repeat(channels, 1, 1, 1))
    
    def _create_sinc_kernel(self, kernel_size, stride):
        """
        Create a sinc kernel (low-pass filter) with Hanning window.
        
        Args:
            kernel_size (int): Size of the kernel
            stride (int): Stride value
            
        Returns:
            torch.Tensor: Sinc kernel
        """
        # Center coordinates
        center = kernel_size // 2
        
        # Create coordinate grid
        y, x = torch.meshgrid(
            torch.arange(kernel_size, dtype=torch.float32),
            torch.arange(kernel_size, dtype=torch.float32)
        )
        
        # Adjust coordinates
        x = x - center
        y = y - center
        
        # Compute radial distance
        r = torch.sqrt(x**2 + y**2)
        
        # Create sinc filter (normalized to cutoff frequency)
        r = torch.where(r == 0, torch.tensor(1e-10), r)
        cutoff = 1.0 / stride
        sinc = torch.sin(math.pi * cutoff * r) / (math.pi * cutoff * r)
        
        # Apply Hanning window
        n = torch.arange(kernel_size)
        hann_window = 0.5 - 0.5 * torch.cos(2 * math.pi * n / (kernel_size - 1))
        hann_2d = torch.outer(hann_window, hann_window)
        
        # Final filter
        kernel = sinc * hann_2d
        
        # Normalize
        kernel = kernel / kernel.sum()
        
        # Reshape to (1, 1, K, K) format
        return kernel.view(1, 1, kernel_size, kernel_size)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Downsampled tensor
        """
        # Apply anti-aliasing filter and downsample
        padding = self.kernel_size // 2
        return F.conv2d(x, self.kernel, stride=self.stride, padding=padding, groups=self.channels)


class TIPSPooling(nn.Module):
    """
    Translation Invariant Polyphase Sampling Pooling.
    
    This module implements a shift-invariant alternative to max pooling or average pooling.
    
    Args:
        channels (int): Number of input channels
        kernel_size (int): Pooling kernel size (default: 2)
        stride (int): Stride for downsampling (default: 2)
        mode (str): Pooling mode ('avg' or 'max') (default: 'avg')
    """
    def __init__(self, channels: int, kernel_size: int = 2, stride: int = 2, mode: str = 'avg'):
        super(TIPSPooling, self).__init__()
        
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.mode = mode
        
        # Create phase-specific pooling operations
        self.phases = stride * stride
        
        # Anti-aliasing filter
        self.aa_filter = TIPSDownsampler(channels, stride=stride, kernel_size=3)
        
        # Ensure we're using correct pooling type
        if mode == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        elif mode == 'max':
            self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        else:
            raise ValueError(f"Unknown pooling mode: {mode}")
        
        # Learnable phase weights
        self.phase_weights = nn.Parameter(torch.ones(self.phases) / self.phases)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Pooled tensor
        """
        # For stride=1, just apply regular pooling
        if self.stride == 1:
            return self.pool(x)
        
        # Apply pooling and get phase outputs
        batch_size, channels, h, w = x.shape
        all_phases = []
        
        # For each phase
        phase_idx = 0
        for i in range(self.stride):
            for j in range(self.stride):
                # Apply phase-specific shift
                offset_h = i / self.stride
                offset_w = j / self.stride
                
                # Shift the input
                shifted_x = torch.roll(x, shifts=(-i, -j), dims=(2, 3))
                
                # Apply pooling
                pooled = self.pool(shifted_x)
                
                # Apply phase weight
                pooled = pooled * self.phase_weights[phase_idx]
                
                all_phases.append(pooled)
                phase_idx += 1
        
        # Sum all phase outputs
        y = torch.stack(all_phases).sum(dim=0)
        
        # Apply anti-aliasing filter and downsample
        y = self.aa_filter(y)
        
        return y


def convert_to_tips_model(model, mode='conv'):
    """
    Convert a standard CNN model to a TIPS version.
    
    This function replaces strided convolutions and/or pooling layers
    with their TIPS versions for better shift invariance.
    
    Args:
        model (nn.Module): The original model
        mode (str): Conversion mode ('conv', 'pool', or 'both') (default: 'conv')
        
    Returns:
        nn.Module: The modified TIPS model
    """
    for name, module in model.named_children():
        # Recursively convert children
        if len(list(module.children())) > 0:
            convert_to_tips_model(module, mode=mode)
            
        # Replace strided convolution
        if (mode == 'conv' or mode == 'both') and isinstance(module, nn.Conv2d) and \
           (module.stride[0] > 1 or module.stride[1] > 1):
            # Get original parameters
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size[0]
            stride = module.stride
            padding = module.padding[0]
            bias = module.bias is not None
            
            # Create TIPS version
            new_conv = TIPSConv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=bias
            )
            
            # Replace the module
            setattr(model, name, new_conv)
            
        # Replace pooling
        if (mode == 'pool' or mode == 'both') and (isinstance(module, nn.MaxPool2d) or 
                                                 isinstance(module, nn.AvgPool2d)):
            # Get original parameters
            kernel_size = module.kernel_size
            stride = module.stride
            
            # Determine pooling mode
            pool_mode = 'max' if isinstance(module, nn.MaxPool2d) else 'avg'
            
            # For the first pooling layer, try to determine the number of channels
            # from previous layer if possible
            if hasattr(model, f"{name.replace('pool', 'conv')}"):
                channels = getattr(model, name.replace('pool', 'conv')).out_channels
            else:
                # Default guess based on common architectures
                channels = 64  # Common number of channels after first conv
            
            # Create TIPS version
            new_pool = TIPSPooling(
                channels, kernel_size=kernel_size, 
                stride=stride, mode=pool_mode
            )
            
            # Replace the module
            setattr(model, name, new_pool)
    
    return model 