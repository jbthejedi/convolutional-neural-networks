import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt

from dataclasses import dataclass
from tqdm import tqdm
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset, random_split

import nn_modules as mynn
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 1337
torch.manual_seed(seed)
print(f"Device {device}")
print(f"Seed {seed}")

@dataclass
class Config:
    normalize_shape : tuple = (0.5, 0.5, 0.5)
    batch_size : int = 32
    n_epochs : int = 1
    image_size : int = 32
    train_split_p : float = 0.9

class BasicBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels,
        kernel_size, stride, padding, downsample=None
    ):
        """
        identity = x
        conv1 (k=3, s=stride, p=1)
        bn1
        relu

        conv1 (k=3, s=1, p=1)
        bn2

        x += identity
        """
        super().__init__()
        self.conv1 = mynn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = mynn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

        self.conv2 = mynn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1
        )
        self.bn2 = mynn.BatchNorm2d(num_features=out_channels)

        if stride !=1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                mynn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, padding=0
                ),
                mynn.BatchNorm2d(num_features=out_channels)
            )

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity

        return x

class Resnet18(nn.Module):
    def __init__(self, num_classes=10):
        """
        conv1 (k=7, s=2, p=3) ->
        bn1 ->
        relu ->
        maxpool ->

        layer 1 -> layer 4 ->

        avgpool -> 
        flatten torch.flatten(x, dim=1) ->
        Linear ->
        out
        """
        super().__init__()
        self.conv1 = mynn.Conv2d(
            in_channels=64, out_channels=64,
            kernel_size=7, stride=2, padding=3
        )
        self.bn1 = mynn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU()
        self.maxpool = mynn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(in_channels=64, out_channels=64, n_blocks=2, stride=1)
        # self.layer1 = self._make_layer(in_channels=64, out_channels=128, n_blocks, stride=2)
        # self.layer1 = self._make_layer(in_channels, out_channels, n_blocks, stride=2)
        # self.layer1 = self._make_layer(in_channels, out_channels, n_blocks, stride=2)

        self.avgpool = mynn.AvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, n_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride=stride))
        for _ in range(1, n_blocks + 1):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, dim=1)
        x = self.fc(x)
        return x

def train_test_model(config : Config):
    # Get full dataset
    dataset = CIFAR10(
        root="~/projects/vision-transformer/data",
        download=False,
        transform=T.Compose([
            T.ToTensor(),
            T.Normalize(config.normalize_shape, config.normalize_shape)
        ])
    )

    # Get sample for testing
    indices = random.sample(range(len(dataset)), 1000)
    dataset = Subset(dataset, indices)

    train_split = int(config.train_split_p * len(dataset))
    train, test = random_split(dataset, [train_split, len(dataset) - train_split])
    traindl = DataLoader(train, batch_size=config.batch_size, shuffle=True)
    testdl = DataLoader(test, batch_size=config.batch_size, shuffle=False)

    model = Resnet18()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(config.n_epochs + 1):
        tqdm.write(f"Epoch {epoch}/{config.n_epochs}")
        model.train()
        with tqdm(desc='Training') as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                break
        break

def main():
    config = Config()
    train_test_model(config)

if __name__ == '__main__':
    main()

























