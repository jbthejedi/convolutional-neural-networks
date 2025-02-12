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
        x_unfold = F.unfold( # (b, l, c_in*k*k)
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

class MyMaxPool2d(nn.Module):
    def __init__(
        self, kernel_size, stride, padding
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        b, c_in, h_in, w_in = x.shape
        k, s, p = self.kernel_size, self.stride, self.padding
        x_unfold = F.unfold( # (b, c_in*k*k, h_out*w_out)
            x, kernel_size=k, stride=s, padding=p
        )
        x_unfold = x_unfold.view(b, c_in, k*k, -1) # (b, c_in, k*k, h_out*w_out)
        x_max, _ = x_unfold.max(dim=2) # (b, c_in, h_out*w_out)
        h_out = (h_in + 2*p - k) // s + 1
        w_out = (w_in + 2*p - k) // s + 1
        out = x_max.view(b, c_in, h_out, w_out)
        return out

class MyAvgPool2d(nn.Module):
    def __init__(self, output_size : tuple):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        b, c_in, h_in, w_in = x.shape

        h_out, w_out = self.output_size[0], self.output_size[1]
        kernel_h = h_in // h_out
        kernel_w = w_in // w_out
        kernel_size = (kernel_h, kernel_w) 
        
        x_unfold = F.unfold( # (b, c_in*k*k, h_out*w_out)
            x, kernel_size=kernel_size, stride=kernel_size, # padding=0
        )
        x_unfold = x_unfold.view(b, c_in, kernel_h * kernel_w, h_out, w_out)
        out = x_unfold.mean(dim=2) # (b, c_in, h_out, w_out)

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

    # --------------
    # MyMaxPool2d
    # --------------
    module_name = "MyMaxPool2d"
    b, c_in, h_in, w_in = (6, 3, 32, 32)
    module = MyMaxPool2d(
        kernel_size=3,
        stride=1,
        padding=1,
    )
    print(f"Testing module {module_name}")
    in_tensor = torch.zeros(b, c_in, h_in, w_in)

    # in (6, 3, 32, 32) => out (6, 20, 32, 32)
    out_tensor = module(in_tensor)
    expected_shape = (6, c_in, h_in, w_in)
    assert out_tensor.shape == expected_shape, f"Failed. Expected {expected_shape} but got {out_tensor.shape}"
    print(f"out_tensor.shape {out_tensor.shape}")

    # --------------
    # MyAvgPool2d
    # --------------
    module_name = "MyAvgPool2d"
    b, c_in, h_in, w_in = (6, 3, 32, 32)
    output_size = (1,1)
    module = MyAvgPool2d(output_size=output_size)
    print(f"Testing module {module_name}")
    in_tensor = torch.zeros(b, c_in, h_in, w_in)

    # in (6, 3, 32, 32) => out (6, 20, 32, 32)
    out_tensor = module(in_tensor)
    expected_shape = (6, c_in, output_size[0], output_size[1])
    assert out_tensor.shape == expected_shape, f"Failed. Expected {expected_shape} but got {out_tensor.shape}"
    print(f"out_tensor.shape {out_tensor.shape}")

def main():
    config = Config()
    test_modules(config)

if __name__ == '__main__':
    main()
