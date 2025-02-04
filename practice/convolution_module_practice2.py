import torch
import torch.nn.functional as F

from torch import nn
from dataclasses import dataclass

@dataclass
class Config:
    out_channels : int = 6
    in_channels : int = 3
    batch_size : int = 32
    kernel_size : int = 2
    padding : int = 1
    stride : int = 1
    bias : bool = True
    
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
        ) if self.config.bias else None

    def forward(self, x):
        B, C_in, H, W = x.shape
        x_unfold = F.unfold(
            x,
            kernel_size=self.config.kernel_size,
            padding=self.config.padding,
            stride=self.config.stride,
        ) # (B, C_in*k^2, L), L = H_out * W_out
        x_unfold_t = x_unfold.transpose(2, 1) # (B, L, C_in*k^2)
        weight = self.weight.view(self.config.out_channels, -1) # (C_out, C_in*k^2)
        out = x_unfold_t @ weight.t() # (B, L, C_out)
        out = out.transpose(1, 2)

        if self.bias is not None:
            out += self.bias.view(1, self.config.out_channels, 1)
            # out += self.bias.view(1, -1, 1)
            
        H_out = (H + 2*self.config.padding - self.config.kernel_size) // self.config.stride + 1
        W_out = (W + 2*self.config.padding - self.config.kernel_size) // self.config.stride + 1
        
        out = out.view(B, self.config.out_channels, H_out, W_out)
        
        return out

# -----------------------
# Testing the custom conv
# -----------------------
if __name__ == "__main__":
    config = Config()
    x = torch.randn(2, 3, 32, 32)  # (B=2, C=3, H=32, W=32)
    my_conv = MyConv2d(config)
    out = my_conv(x)
    print("Output shape from MyConv2d:", out.shape)
    
    config = Config()
    config.stride = 2
    config.kernel = 2
    config.padding = 0
    x = torch.randn(2, 3, 32, 32)  # (B=2, C=3, H=32, W=32)
    my_conv = MyMaxPool2d(config)
    out = my_conv(x)
    print("Output shape from MyMaxPool2d:", out.shape)
    
    # # Compare with built-in conv for correctness
    # torch_conv = nn.Conv2d(
    #     in_channels=3,
    #     out_channels=6,
    #     kernel_size=3,
    #     stride=1,
    #     padding=1,
    #     bias=True
    # )
    # out_torch = torch_conv(x)
    # print("Output shape from torch.nn.Conv2d:", out_torch.shape)
