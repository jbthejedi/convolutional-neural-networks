import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from dataclasses import dataclass
from convolution2d import MyConv2d

@dataclass
class TrainingConfig:
    batch_size : int = 32
    n_epochs : int = 1
    image_size : int = 32
    normalize_shape : tuple = ((0.5,), (0.5,), (0.5,))
    train_split : float = 0.8

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        downsample=None
    ):
        self.conv1 = MyConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = MyConv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample


    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample:
            identity = self.downsample(identity)

        x = x + identity
        x = self.relu(x)

        return x

class MyResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        self.current_channels = 64

        self.conv1 = MyConv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3, # same padding (k-1)//3
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # lowers spatial dimensions

        # self.layer1 = self.

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.current_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                MyConv2d(
                    in_channels=self.current_channels,
                    out_channels=self.out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.current_channels, out_channels, stride, downsample))
        self.current_channels = out_channels * block.expansion
        # for _ in range(1, num_blocks):
        layers.append(block(self.current_channels, out_channels))
        
        return nn.Sequential(*layers)


    def _init_weights(self):
        pass

    def forward(self, x):
        return x

def train_test_model(config : TrainingConfig):
    dataset = CIFAR10(
        root="../../vision-transformer",
        transform=T.Compose([
            T.Resize((config.image_size, config.image_size))
    ])
    )
    model = MyResNet(BasicBlock, layers=[2,2,2,2], num_classes=10)

def main():
    config = TrainingConfig()
    train_test_model(config)

if __name__ == '__main__':
    main()
