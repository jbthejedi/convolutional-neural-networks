import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

seed = 1337
torch.manual_seed(seed)

class MyConv2D(nn.Module):
    def __init__(
        self, in_channels, out_channels,
        kernel_size, stride, padding
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        k = kernel_size
        self.weights = nn.Parameter( # (C_out, C, k, k)
            torch.randn((out_channels, in_channels, k, k))
        )

    def forward(self, x):
        B, c_in, h_in, w_in = x.shape
        x_unfold = F.unfold( # (B, C*k*k, L), L = H_out * W_out
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )
        x_unfold_t = x_unfold.transpose(1, 2) # (B, L, C*k*k)
        c_out = self.out_channels
        k = self.kernel_size
        weights_reshaped = self.weights.view(c_out, c_in * k * k)
        out = x_unfold_t @ weights_reshaped.t() # (B, L, C_out)
        h_out = (h_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (w_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.view(B, -1, h_out, w_out)

        return out

class MyMaxPool2D(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        """
        out.shape = (b, c_in, h_out, w_out)
        """
        b, c_in, h_in, w_in = x.shape
        k, s, p = self.kernel_size, self.stride, self.padding
        x_unfold = F.unfold( # (b, c_in*k*k, L), where L = h_out * w_out
            x, kernel_size=k, stride=s, padding=p
        )
        # get to (b, c_in, L)
        x_unfold = x_unfold.view(b, c_in, k*k, -1) # (b, c, k*k, L)
        x_max, _ = x_unfold.max(dim=2) # (b, c, L)
        h_out = (h_in + 2 * p - k) // s + 1
        w_out = (w_in + 2 * p - k) // s + 1
        out = x_max.view(b, c_in, h_out, w_out)

        return out

class MyAveragePool2D(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        b, c_in, h_in, w_in = x.shape
        print(f"h_in {h_in} w_in {w_in}")
        h_out, w_out = self.output_size
        kernel_size = (h_in // h_out, w_in // w_out)
        x_unfold = F.unfold( # (B, c_in*k*k, L)
            x, kernel_size=kernel_size, stride=kernel_size # padding=0 default
        )
        x_unfold = x_unfold.view(b, c_in, kernel_size[0]*kernel_size[1], h_out, w_out) # (b, c_in, k*k, h_out, w_out)
        out = x_unfold.mean(dim=2) # (b, c_in, h_out, w_out)

        return out

class LayerNorm:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def __call__(self, x):
        xmean = x.mean(-1, keepdim=True)
        xvar = x.var(-1, keepdim=True)
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        # print(f"ln.shape {self.out.shape}")
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class MyBatchNorm2D(nn.Module):
    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
        ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.running_mean = None
            self.running_var = None

    def forward(self, x):
        if self.training:
            batch_mean = x.mean(dim=[0, 2, 3])
            batch_var = x.var(dim=[0, 2, 3], unbiased=False)

            if self.track_running_stats:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_norm = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        if self.affine:
            x_norm = self.gamma[None, :, None, None] * x_norm + self.beta[None, :, None, None]
        return x

def test_module():
    # ------------
    # Convolution
    # ------------
    in_tensor = torch.randn((4, 3, 16, 16))
    module = MyConv2D(
        in_channels=3,
        out_channels=12,
        kernel_size=3,
        stride=1,
        padding=1
    )
    h_in, k, s, p = 16, 3, 1, 1
    h_out = (h_in + 2*p - k) // s + 1
    w_out = h_out
    out_tensor = module(in_tensor)

    assert out_tensor.shape == (4, 12, h_out, w_out)

    # ------------
    # MaxPool2D
    # ------------
    in_tensor = torch.randn((4, 3, 16, 16))
    module = MyMaxPool2D(
        kernel_size=3,
        stride=1,
        padding=1
    )
    h_in, k, s, p = 16, 3, 1, 1
    h_out = (h_in + 2*p - k) // s + 1
    w_out = h_out
    out_tensor = module(in_tensor)
    print(out_tensor.shape)

    assert out_tensor.shape == (4, 3, h_out, w_out)

    # ------------
    # AveragePool2d
    # ------------
    in_tensor = torch.randn((4, 3, 16, 16))
    module = MyAveragePool2D(output_size=(1, 1))
    h_in, k, s, p = 16, 3, 1, 1
    h_out = (h_in + 2*p - k) // s + 1
    w_out = h_out
    out_tensor = module(in_tensor)
    print(out_tensor.shape)

    assert out_tensor.shape == (4, 3, 1, 1), "failed"

    # ------------
    # MyBatchNorm2D 
    # ------------
    in_tensor = torch.randn((4, 3, 32, 32))
    module = MyBatchNorm2D(3)
    h_in, k, s, p = 16, 3, 32, 32
    h_out = (h_in + 2*p - k) // s + 1
    w_out = h_out
    out_tensor = module(in_tensor)
    print(out_tensor.shape)

    assert out_tensor.shape == (4, 3, 32, 32), "failed"

def main():
    test_module()

if __name__ == '__main__':
    main()
