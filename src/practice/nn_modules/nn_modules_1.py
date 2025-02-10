import torch
import torch.nn as nn

class MyConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels,
        kernel_size, stride, padding
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # (c_out, c_in*k*k)
        self.weights = nn.Parameter(
            torch.randn(
                self.out_channels, self.in_channels,
                self.kernel_size, self.kernel_size
            )
        )
