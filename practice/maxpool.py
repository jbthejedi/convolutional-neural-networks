import torch
import torch.nn.functional as F

from torch import nn
from dataclasses import dataclass

class MyMaxPool2d(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x):
        B, C, H_in, W_in = x.shape
        k = self.config.kernel_size
        
        x_unfold = F.unfold( # (B, C*k^2, L), where L = H_out*W_out
            x,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding
        )
        x = x.view(B, C, k*k, -1)
        print(f"x.shape {x.shape}")
        x_max, _ = x.max(dim=2) # (B, C, L)
        H_out = (H_in + 2*self.config.padding - self.config.kernel_size) // self.config.stride + 1
        W_out = (W_in + 2*self.config.padding - self.config.kernel_size) // self.config.stride + 1
        out = x_max.view(B, C, H_out, W_out)
        
        return out
