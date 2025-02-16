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
    def __init__(self, kernel_size, stride, padding):
        super().__init__()

class Resnet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

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

























