import torch
import torch.nn.functional as F

from torch import nn
from dataclasses import dataclass

@dataclass
class Config:
    kernel_size        : int   = 3
    stride             : int   = 1
    padding            : int   = 1
    in_channels        : int   = 3
    out_channels       : int   = 6
    batch_size         : int   = 8
    weight_scale       : float = 0.01
    bias               : bool  = True
    
class MyConv2d(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # weight.shape = (C_out, C_in, k, k)
        self.weight = nn.Parameter(
            torch.randn(
                config.out_channels,
                config.in_channels,
                config.kernel_size,
                config.kernel_size,
            ) * config.weight_scale
        )

        self.bias = nn.Parameter(
            torch.zeros(config.out_channels)
        ) if config.bias else None

    def forward(self, x):
        """
        x.shape = (B, C_in, H_in, W_in)
        H_out = H_in + 2p - k // s + 1
        intermediate shape = (B, L, C_in*k*k), where
        # L is the num of patches extracted, L = H_out * W_out
        out.shape = (B, C_out, H_out, W_out)
        """
        B, C_in, H_in, W_in = x.shape
        x_unfold = F.unfold( # (B, C_in*k^2, L)
            x,
            kernel_size=self.config.kernel_size,
            padding=self.config.padding,
            stride=self.config.stride,
        )
        x_unfold_t = x_unfold.transpose(1, 2) # (B, L, C_in*k^2)
        print(f"x_unfold_t.shape {x_unfold_t.shape}")
        weight = self.weight.view(self.config.out_channels, -1) # (C_out, C_in*k^2)
        print(f"weight.t().shape {weight.t().shape}")
        A = x_unfold_t @ weight.t() # (B, L, C_out)
        A = A.transpose(2, 1) # (B, C_out, L)
        B, C_out, L = A.shape
        print(f"A.shape {A.shape}")

        H_out = (H_in + 2*self.config.padding - self.config.kernel_size // self.config.stride) + 1
        W_out = (W_in + 2*self.config.padding - self.config.kernel_size // self.config.stride) + 1
        out = A.view(B, C_out, H_out, W_out)
        return out
        
# -----------------------
# Testing the custom conv
# -----------------------
if __name__ == "__main__":
    config = Config()
    
    # Sample input
    x = torch.randn(2, 3, 32, 32)  # (B=2, C=3, H=32, W=32)

    # Our custom conv
    my_conv = MyConv2d(config)
    
    # Forward pass
    out = my_conv(x)
    print("Output shape from MyConv2d:", out.shape)
    
    # Compare with built-in conv for correctness
    torch_conv = nn.Conv2d(
        in_channels=3,
        out_channels=6,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True
    )
    out_torch = torch_conv(x)
    print("Output shape from torch.nn.Conv2d:", out_torch.shape)
