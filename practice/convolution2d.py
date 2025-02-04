import torch
import torch.nn as nn
import torch.nn.functional as F

class MyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_flag = bias  # store bias flag if needed

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        batch_size, _, height, width = x.shape
        # Extract sliding local blocks
        x_unfold = F.unfold(
            x,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        )  # Shape: (B, C_in * K*K, L)
        # Reshape to (B, L, C_in * K*K)
        x_unfold = x_unfold.transpose(1, 2)
        # Reshape weights to (C_out, C_in * K*K)
        weight_reshaped = self.weight.view(self.out_channels, -1)
        # Matrix multiplication (convolution)
        out = x_unfold @ weight_reshaped.t()  # Shape: (B, L, C_out)
        out = out.transpose(1, 2)  # Shape: (B, C_out, L)

        if self.bias is not None:
            out += self.bias.view(1, self.out_channels, 1)

        # Compute output spatial dimensions
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.view(batch_size, self.out_channels, out_height, out_width)
        return out

