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
    padding               =1 # "same" padding for kernel_size=3
    bias: bool            = True


class MyConv2d(nn.Module):
    def __init__(self, config : Config):
        super().__init__()
        
        # Store hyperparams
        self.config = config
        
        # Weight and bias parameters
        # shape: (out_channels, in_channels, kernel_size, kernel_size)
        # Num_params = out_channelsxin_channels x kernel_size x kernel_size
        self.weight = nn.Parameter(
            torch.randn(
                config.out_channels,
                config.in_channels,
                config.kernel_size,
                config.kernel_size
            ) * 0.01
        )
        
        # optional bias
        if config.bias:
            config.bias = nn.Parameter(torch.zeros(config.out_channels))
        else:
            config.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (B, in_channels, H, W)
        Returns: (B, out_channels, H_out, W_out)
        """
        B, C, H, W = x.shape
        assert C == self.config.in_channels, \
            f"Expected in_channels={self.config.in_channels}, got {C}"

        # -----------------------------
        # 1) Use unfold to get patches
        # -----------------------------
        #  -> x_unfold shape: (B, in_channels * kernel_size * kernel_size, L)
        #     where L = number_of_patches = H_out * W_out
        x_unfold = F.unfold(
            x, 
            kernel_size=self.config.kernel_size,
            padding=self.config.padding,
            stride=self.config.stride
        )
        # x = x.transpose(1, 2) # (B, L, C*k*k) is more intuitive to me
        
        # -----------------------------
        # 2) Multiply + sum over patch
        # -----------------------------
        # We want to do a matrix multiply between:
        #   weights: (out_channels, in_channels * kernel_size * kernel_size)
        #   x_unfold: (B, in_channels * kernel_size * kernel_size, L)
        # to get output shape (B, out_channels, L).
        
        # Reshape weight for matmul: (out_channels, in_channels * kernel_size^2)
        weight = self.weight.view(self.config.out_channels, -1)
        
        # x_unfold is (B, C * K * K, L) but we want to batch matmul from the perspective of B:
        # One approach: 
        #   - Transpose x_unfold to (B, L, C*K*K)
        #   - Then we can do (out_channels, C*K*K) @ (C*K*K, L) for each B
        #   - We'll reshape to (B, out_channels, L).
        
        x_unfold_t = x_unfold.transpose(1, 2)  # (B, L, C*K*K)
        out = x_unfold_t.matmul(weight.t())    # (B, L, out_channels)
        out = out.transpose(1, 2)              # (B, out_channels, L)

        # --------------------------------
        # 3) Add bias (if it exists)
        # --------------------------------
        if self.config.bias is not None:
            # broadcast bias over the second dimension
            out += self.config.bias.view(1, -1, 1)
        
        # -----------------------------
        # 4) Reshape to (B, out_channels, H_out, W_out)
        # -----------------------------
        # L = H_out * W_out, we can compute H_out and W_out from the unfold operation
        # But we can also deduce them from the stride/padding/kernel_size formula:
        
        H_out = (H + 2*self.config.padding - self.config.kernel_size) // self.config.stride + 1
        W_out = (W + 2*self.config.padding - self.config.kernel_size) // self.config.stride + 1
        
        out = out.view(B, self.config.out_channels, H_out, W_out)
        
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
