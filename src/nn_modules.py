import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

@dataclass
class Config:
    batch_size : int = 32
    weight_init_scale : float = 0.02

class MyConv2d(nn.Module):
    """
    ioksp: c_in, c_out, kern, strid, pad
    h_out = (h_in + 2*p - k) // s + 1
    """
    def __init__(
        self, in_channels, out_channels,
        kernel_size, stride, padding, weight_init_scale
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        scale = weight_init_scale
        # weights.shape => (c_out, c_in, k, k)
        self.weights = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        )

    def forward(self, x):
        """
        in.shape (b, c_in, h_in, w_in)
        return shape (b, c_out, h_out, w_out)
        where h_out = (h_in + 2p - k) // s + 1
        """
        b, c_in, h_in, w_in = x.shape
        k, s, p = self.kernel_size, self.stride, self.padding
        x_unfold = F.unfold( # (b, c_in*k*k, l), where l = h_out * w_out
            x, kernel_size=k, stride=s, padding=p
        )
        x_unfold_t = x_unfold.transpose(1, 2) # (b, l, c_in*k*k)
        weights = self.weights.view(-1, c_in*k*k) # (c_out, c_in*k*k)
        out = x_unfold_t @ weights.t() # (b, l, c_in*k*k) x (c_in*k*k, c_out) => (b, l, c_out)
        h_out = (h_in + 2 * p - k ) // s + 1
        w_out = (w_in + 2 * p - k ) // s + 1
        out = out.view(b, -1, h_out, w_out)

        return out

class MyMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        b, c_in, h_in, w_in = x.shape
        k, s, p = self.kernel_size, self.stride, self.padding
        x_unfold = F.unfold( # (b, c_in*k*k, l), where l = h_out*w_out
            x, kernel_size=k, stride=s, padding=p
        )
        x_unfold = x_unfold.view(b, c_in, k*k, -1) # (b, c_in, k*k, l)
        xmax, _ = x_unfold.max(dim=2) # (b, c_in, l)
        h_out = (h_in + 2*p - k) // s + 1
        w_out = (w_in + 2*p - k) // s + 1
        out = xmax.view(b, c_in, h_out, w_out)

        return out

class MyAvgPool2d(nn.Module):
    def __init__(self, output_size : tuple):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        b, c_in, h_in, w_in = x.shape
        h_out, w_out = self.output_size
        kernel_h = h_in // h_out
        kernel_w = w_in // w_out
        kernel_size = (kernel_h, kernel_w)
        x_unfold = F.unfold( # (b, c_in*k*k, l)
            x, kernel_size=kernel_size, stride=kernel_size
        )
        x_unfold = x_unfold.view(b, c_in, kernel_h * kernel_w, -1)
        xmean = x_unfold.mean(dim=2) # (b, c_in, l)
        out = xmean.view(b, c_in, h_out, w_out)
        return out

def test_modules(config : Config):
    # ----------
    # MyConv2d
    # ----------
    module_name = "MyConv2d"
    module = MyConv2d(
        in_channels=3,
        out_channels=20,
        kernel_size=3,
        stride=1,
        padding=1,
        weight_init_scale=config.weight_init_scale,
    )

    in_tensor = torch.randn((6, 3, 32, 32))
    out_tensor = module(in_tensor)
    expected_shape = (6, 20, 32, 32)
    assert out_tensor.shape == expected_shape, f"Failed. Expected {expected_shape} but got {out_tensor.shape}"
    print(f"out_tensor.shape {out_tensor.shape}")

    # ----------
    # MyMaxPool2d
    # ----------
    module_name = "MyMaxPool2d"
    module = MyMaxPool2d(
        kernel_size=3,
        stride=1,
        padding=1,
    )

    in_tensor = torch.randn((6, 3, 32, 32))
    out_tensor = module(in_tensor)
    expected_shape = (6, 3, 32, 32)
    assert out_tensor.shape == expected_shape, f"Failed. Expected {expected_shape} but got {out_tensor.shape}"
    print(f"out_tensor.shape {out_tensor.shape}")

    # ----------
    # MyAvgPool2d
    # ----------
    module_name = "MyAvgPool2d"
    module = MyAvgPool2d(output_size=(1, 1))

    in_tensor = torch.randn((6, 20, 8, 8)) 
    out_tensor = module(in_tensor)
    expected_shape = (6, 20, 1, 1)
    assert out_tensor.shape == expected_shape, f"Failed. Expected {expected_shape} but got {out_tensor.shape}"
    print(f"out_tensor.shape {out_tensor.shape}")


def main():
    config = Config()
    test_modules(config)

if __name__ == '__main__':
    main()
