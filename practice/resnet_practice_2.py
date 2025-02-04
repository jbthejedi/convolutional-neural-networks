import torch
import torch.nn.functional as F
import torchvision.transforms as T

from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from dataclasses import dataclass

seed = 1337
torch.manual_seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"SEED {seed}")
print(f"DEVICE {device}")

@dataclass
class ConvConfig:
    in_channels    : int
    out_channels   : int
    stride         : int
    kernel_size    : int
    padding        : int
    bias           : bool
    
def make_conv_config(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
    return ConvConfig(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias
    )

@dataclass
class TrainingConfig:
    num_classes: int = 10
    n_epochs: int = 10
    p_train_split: float = 0.8
    image_size: int = 32
    batch_size: int = 32

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
        self.bias = (nn.Parameter(torch.zeros(config.out_channels))
                     if self.config.bias else None)

    def forward(self, x):
        B, C_in, H, W = x.shape
        x_unfold = F.unfold(
            x,
            kernel_size=self.config.kernel_size,
            padding=self.config.padding,
            stride=self.config.stride,
        )  # (B, C_in*k^2, L)
        x_unfold_t = x_unfold.transpose(2, 1)  # (B, L, C_in*k^2)

        # (C_out, C_in*k^2)
        weight = self.weight.view(self.config.out_channels, -1)
        
        # (B, L, C_out)
        out = x_unfold_t @ weight.t()
        
        # (B, C_out, L)
        out = out.transpose(1, 2)

        if self.bias is not None:
            out += self.bias.view(1, self.config.out_channels, 1)

        # Compute output spatial size
        H_out = (H + 2*self.config.padding - self.config.kernel_size) // self.config.stride + 1
        W_out = (W + 2*self.config.padding - self.config.kernel_size) // self.config.stride + 1
        
        # (B, C_out, H_out, W_out)
        out = out.view(B, self.config.out_channels, H_out, W_out)
        
        return out

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = MyConv2d(
            make_conv_config(
                in_channels=in_planes,
                out_channels=planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = MyConv2d(
            make_conv_config(
                in_channels=planes,
                out_channels=planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            )
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(x)
        out = self.relu(x)

        out = self.conv2(x)
        out = self.bn2(x)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)

        return out

class MyResNet(nn.Module):
    def __init__(self, block_type, layers, num_classes=1000):
        super().__init__()
        self.in_planes = 64
        self.conv1 = MyConv2d(
            make_conv_config(
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.layer1 = self._make_layer(
            block_type=block_type,
            out_planes=64,
            n_layers=layers[0],
            stride=1
        )
        self.layer2 = self._make_layer(
            block_type=block_type,
            out_planes=128,
            n_layers=layers[1],
            stride=2,
        )
        self.layer3 = self._make_layer(
            block_type=block_type,
            out_planes=256,
            n_layers=layers[2],
            stride=2,
        )
        self.layer4 = self._make_layer(
            block_type=block_type,
            out_planes=512,
            n_layers=layers[3],
            stride=2,
        )

        self.avgpool = nn.AdaptiveAvgPool2d(()
        self.fc = nn.Linear(512 * block_type.expansion, num_classes)

    def _make_layer(self, block_type, out_planes, n_layers, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != 1 * block.expansion:
            downsample = nn.Sequential(
            )

        layers = []
        layers.append(block_type(self.in_planes, out_planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, n_layers):
            layers.append(block_type(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x 
        
def main():
    config = TrainingConfig()
    model = MyResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=config.num_classes
    )
    
    
if __name__ == '__main__':
    main()
