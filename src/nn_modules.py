import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

@dataclass
class Config:
    weight_init_scale : float = 0.02
    batch_size : int = 8

# ioksp
class MyConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels,
        kernel_size, stride, padding, weight_scale
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        s = weight_scale
        self.weights = nn.Parameter( # (c_out, c_in, k, k) => (c_out, c_in*k*k)
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * s
        )

    def forward(self, x):
        """
        in shape (b, c_in, h_in, w_in)
        return shape (b, c_out, h_out, w_out)
        """
        b, c_in, h_in, w_in = x.shape
        k, s, p = self.kernel_size, self.stride, self.padding

        # unfold => (b, c_in*k*k, l), where l = h_out * w_out is
        # the number of extracted patches.
        x_unfold = torch.nn.functional.unfold( # (b, l, c_in*k*k)
                x, kernel_size=k, stride=s, padding=p
        )
        x_unfold_t = x_unfold.transpose(1, 2) # (b, l, c_in*k*k)
        weights = self.weights.view(-1, c_in*k*k) # (c_out, c_in*k*k)
        out = x_unfold_t @ weights.t() # (b, l, c_in*k*k) @ (c_in*k*k, c_out) => (b, l, c_out), where l = (h_out, w_out)
        # Now, we want shape (b, c_out, h_out, w_out)
        h_out = (h_in + 2*p - k) // s + 1
        w_out = (w_in + 2*p - k) // s + 1
        out = out.view(b, -1, h_out, w_out)

        return out

def test_modules(config : Config):
    # --------------
    # MyConv2d
    # --------------
    module_name = "MyConv2d"
    b, c_in, h_in, w_in = (6, 3, 32, 32)
    module = MyConv2d(
        in_channels=c_in,
        out_channels=20,
        kernel_size=3,
        stride=1,
        padding=1,
        weight_scale=config.weight_init_scale,
    )
    print(f"Testing module {module_name}")
    in_tensor = torch.zeros(b, c_in, h_in, w_in)

    # in (6, 3, 32, 32) => out (6, 20, 32, 32)
    out_tensor = module(in_tensor)
    expected_shape = (6, 20, 32, 32)
    assert out_tensor.shape == expected_shape, f"Failed. Expected {expected_shape} but got {out_tensor.shape}"

    print(f"out_tensor.shape {out_tensor.shape}")

def main():
    config = Config()
    test_modules(config)

if __name__ == '__main__':
    main()
