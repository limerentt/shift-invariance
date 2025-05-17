#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implementation of anti-aliasing layers for CNN architectures.
Based on "Making Convolutional Networks Shift-Invariant Again" (Zhang, 2019).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, List, Optional

class BlurPool(nn.Module):
    """
    Anti-aliasing layer with blur kernel.
    
    Implements low-pass filtering with a blur kernel before downsampling
    to prevent aliasing artifacts in CNNs.
    
    Args:
        channels (int): Number of input channels
        kernel_size (int): Size of the blur kernel (2, 3, 5, or 7)
        stride (int): Stride for downsampling (default: 2)
        padding (int or None): Padding, or None for automatic padding
    """
    def __init__(self, channels: int, kernel_size: int = 3, stride: int = 2, padding: Optional[int] = None):
        super(BlurPool, self).__init__()
        
        if kernel_size not in [2, 3, 5, 7]:
            raise ValueError("Kernel size must be one of: 2, 3, 5, 7")
        
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        # Define padding
        self.padding = padding if padding is not None else int((kernel_size - 1) // 2)
        
        # Create filter coefficients
        if kernel_size == 2:
            a = [1.0, 1.0]
        elif kernel_size == 3:
            a = [1.0, 2.0, 1.0]
        elif kernel_size == 5:
            a = [1.0, 4.0, 6.0, 4.0, 1.0]
        elif kernel_size == 7:
            a = [1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0]
            
        # Normalize filter coefficients
        a = torch.tensor(a, dtype=torch.float32)
        a = a / a.sum()
        
        # Create blur kernel (separable 2D filter)
        a = a[None, :] * a[:, None]  # Outer product
        
        # Reshape to (C, 1, K, K) format for depthwise convolution
        self.register_buffer('filter', a[None, None, :, :].repeat(channels, 1, 1, 1))
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Blurred and downsampled tensor
        """
        return F.conv2d(x, self.filter, stride=self.stride, padding=self.padding, groups=self.channels)
    
    def __repr__(self):
        return f"BlurPool(channels={self.channels}, kernel_size={self.kernel_size}, stride={self.stride})"


class AntiAliasDownsample(nn.Module):
    """
    Anti-alias downsampling module.
    
    Combines a standard convolution with blur pooling to create
    a shift-invariant downsampling operation.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Convolution kernel size
        stride (int): Stride for downsampling (must be >= 1)
        padding (int): Padding for convolution
        blur_kernel (int): Size of the blur kernel (default: 3)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 2, padding: int = 1, blur_kernel: int = 3):
        super(AntiAliasDownsample, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                             stride=1, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        if stride == 1:
            self.blur = nn.Identity()
        else:
            self.blur = BlurPool(out_channels, kernel_size=blur_kernel, stride=stride)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Anti-aliased and downsampled tensor
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.blur(x)
        return x


class TIPS(nn.Module):
    """
    Translation Invariant Polyphase Sampling (TIPS) module.
    
    Implements the TIPS method from "TIPS: Translation Invariant Polyphase Sampling" (Saha & Gokhale, 2024).
    
    Args:
        channels (int): Number of input channels
        stride (int): Stride for downsampling (default: 2)
        kernel_size (int): Size of the blur kernel (default: 3)
    """
    def __init__(self, channels: int, stride: int = 2, kernel_size: int = 3):
        super(TIPS, self).__init__()
        
        self.channels = channels
        self.stride = stride
        self.kernel_size = kernel_size
        
        # Create filters for each phase
        self.phases = stride * stride
        
        # Initialize filters for each phase
        filters = []
        for i in range(stride):
            for j in range(stride):
                # Create phase-specific filter
                phase_filter = self._create_phase_filter(i, j, stride, kernel_size)
                filters.append(phase_filter)
        
        # Stack all phase filters into a single tensor
        filters = torch.stack(filters)
        
        # Reshape filters to (phases*C, 1, K, K)
        filters = filters.repeat(channels, 1, 1, 1)
        self.register_buffer('filters', filters)
        
        # Learnable phase weights (initialized equally)
        self.phase_weights = nn.Parameter(torch.ones(self.phases) / self.phases)
        
        # Weight mixer to combine all phase outputs
        self.weight_mixer = nn.Conv2d(channels * self.phases, channels, 
                                     kernel_size=1, groups=channels, bias=False)
        nn.init.constant_(self.weight_mixer.weight, 1.0 / self.phases)
    
    def _create_phase_filter(self, offset_x, offset_y, stride, kernel_size):
        """
        Create a phase-specific filter.
        
        Args:
            offset_x (int): X offset for the phase
            offset_y (int): Y offset for the phase
            stride (int): Stride value
            kernel_size (int): Kernel size
            
        Returns:
            torch.Tensor: Phase-specific filter
        """
        # Center coordinates
        center = kernel_size // 2
        
        # Create coordinate grid
        y, x = torch.meshgrid(
            torch.arange(kernel_size, dtype=torch.float32),
            torch.arange(kernel_size, dtype=torch.float32)
        )
        
        # Adjust coordinates based on phase offset
        x = x - center - offset_x / stride
        y = y - center - offset_y / stride
        
        # Compute radial distance
        r = torch.sqrt(x**2 + y**2)
        
        # Create sinc filter
        r = torch.where(r == 0, torch.tensor(1e-10), r)
        sinc = torch.sin(np.pi * r / stride) / (np.pi * r / stride)
        
        # Apply Hanning window
        n = torch.arange(kernel_size)
        hann_window = 0.5 - 0.5 * torch.cos(2 * np.pi * n / (kernel_size - 1))
        hann_2d = torch.outer(hann_window, hann_window)
        
        # Final filter
        filt = sinc * hann_2d
        
        # Normalize
        return filt / filt.sum()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: TIPS processed tensor
        """
        B, C, H, W = x.shape
        
        # Apply each phase filter
        outputs = []
        for i in range(self.phases):
            # Extract current phase filter
            phase_filter = self.filters[i*C:(i+1)*C]
            
            # Apply filter
            y = F.conv2d(x, phase_filter, stride=self.stride, padding=self.kernel_size//2, groups=C)
            
            # Apply phase weight
            y = y * self.phase_weights[i]
            outputs.append(y)
        
        # Concatenate all phases
        y = torch.cat(outputs, dim=1)
        
        # Mix phase outputs
        y = self.weight_mixer(y)
        
        return y


def convert_to_antialiased_model(model, blur_kernel_size=3):
    """
    Convert a standard CNN model to an anti-aliased version.
    
    This function replaces strided convolutions and max-pooling layers
    with their anti-aliased versions.
    
    Args:
        model (nn.Module): The original model
        blur_kernel_size (int): Size of the blur kernel to use (default: 3)
        
    Returns:
        nn.Module: The modified anti-aliased model
    """
    for name, module in model.named_children():
        # Recursively convert children
        if len(list(module.children())) > 0:
            convert_to_antialiased_model(module, blur_kernel_size)
            
        # Replace strided convolution
        if isinstance(module, nn.Conv2d) and module.stride[0] > 1:
            # Get original parameters
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size[0]
            stride = module.stride[0]
            padding = module.padding[0]
            bias = module.bias is not None
            
            # Create anti-aliased version
            new_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, 
                         stride=1, padding=padding, bias=bias),
                BlurPool(out_channels, kernel_size=blur_kernel_size, stride=stride)
            )
            
            # Replace the module
            setattr(model, name, new_conv)
            
        # Replace max pooling
        elif isinstance(module, nn.MaxPool2d) and module.stride > 1:
            # Get original parameters
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding
            
            # Create anti-aliased version
            new_pool = nn.Sequential(
                nn.MaxPool2d(kernel_size, stride=1, padding=padding),
                BlurPool(module._modules['0'].in_channels if hasattr(module, '_modules') else 
                         getattr(model, name.replace('pool', 'conv')).out_channels,
                         kernel_size=blur_kernel_size, stride=stride)
            )
            
            # Replace the module
            setattr(model, name, new_pool)
    
    return model 