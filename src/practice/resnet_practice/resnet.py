import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

@dataclass
class Config:
    out_channels: int
    in_channels: int
    kernel_size: int
    stride: int
    padding: int
    bias: bool = True
    
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


def make_conv_config(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
    return Config(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias
    )


class BasicBlock(nn.Module):
    expansion = 1  # for BasicBlock, output channels == out_channels

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        # 1st conv
        self.conv1 = MyConv2d(
            make_conv_config(
                in_channels=in_planes, 
                out_channels=planes, 
                kernel_size=3, 
                stride=stride, 
                padding=1, 
                bias=False
            )
        )
        self.bn1 = nn.BatchNorm2d(planes)

        # 2nd conv
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
        self.downsample = downsample  # If needed to match spatial size / channels

    def forward(self, x):
        identity = x

        out = self.conv1(x)      # MyConv2d
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)    # MyConv2d
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out
        
downsample = nn.Sequential(
    MyConv2d(make_conv_config(in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=False)),
    nn.BatchNorm2d(planes)
)

class MyResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        """
        Args:
            block:       the residual block class (BasicBlock or Bottleneck)
            layers:      list containing number of blocks in each layer (e.g. [2,2,2,2] for ResNet-18)
            num_classes: classification output dimension
        """
        super().__init__()
        self.in_planes = 64  # initial #channels after the first conv

        # 1) Initial conv: 7x7, stride=2, padding=3
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

        # 2) MaxPool (3x3, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 3) The main layers (Layer1..Layer4)
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 4) Final linear layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global avg pool
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Optional: weight initialization if desired (PyTorch uses kaiming by default)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, MyConv2d):
                # You might want to set your own initialization here 
                # or rely on the * 0.01 in MyConv2d
                pass
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Create one layer of the ResNet:
          - The first block might have stride>1 to downsample
          - The number of channels changes from self.in_planes to planes
        """
        downsample = None
        # If we need to downsample because of stride > 1 or channel mismatch:
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                MyConv2d(
                    make_conv_config(
                        in_channels=self.in_planes,
                        out_channels=planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        padding=0,
                        bias=False
                    )
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        # subsequent blocks (stride=1)
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Main blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pool & FC
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x