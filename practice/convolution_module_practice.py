import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class Config:
    in_channels: int      = 3
    out_channels: int     = 6
    kernel_size: int      = 3
    stride: int           = 1
    # padding: int          = 0
    padding               = 1 # "same" padding for kernel_size=3
    bias: bool            = True


class MyConv2d(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(
            torch.randn(
                config.out_channels,
                config.in_channels,
                config.kernel_size,
                config.kernel_size,
            ) * 0.01
        )
        self.bias = nn.Parameter(
            torch.zeros(config.out_channels)
        ) if config.bias else None

    def forward(self, x):
        B, C_in, H, W = x.shape
        x_unfold = F.unfold( # (B, C_in*k^2, L), L=H_out*W_out
            x,
            kernel_size=self.config.kernel_size,
            padding=self.config.padding,
            stride=self.config.stride,
        )
        
        weight = self.weight.view(self.config.out_channels, -1) # (C_out, C_in*k^2)
        x_unfold_t = x_unfold.transpose(2, 1) # (B, L, C_in*k^2)
        out = x_unfold_t @ weight.t() # (B, L, C_out)
        out = out.transpose(1, 2) # (B, C_out, L)

        if self.bias is not None:
            out += self.bias.view(1, self.config.out_channels, 1)

        H_out = (H + 2*self.config.padding - self.config.kernel_size) // self.config.stride + 1
        W_out = (W + 2*self.config.padding - self.config.kernel_size) // self.config.stride + 1

        out.view(B, self.config.out_channels, H_out, W_out)
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
