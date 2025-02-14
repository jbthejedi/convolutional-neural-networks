import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split

from dataclasses import dataclass
from tqdm import tqdm

seed = 1337
torch.manual_seed(seed)
device = "cuda" if torch.cuda.is_available() else 'cpu'
print(f"Device {device}")
print(f"Seed {seed}")

@dataclass
class Config:
    batch_size : int = 32
    in_channels : int = 3
    n_epochs : int = 1


def test_modules(config : Config):
    b, c_in, w_in, h_in = (4, 3, 32, 32)
    in_tensor = torch.randn(b, c_in, w_in, h_in)
    print(f"in_tensor.shape {in_tensor.shape}")

def main():
    config = Config()
    test_modules(config)

if __name__ == '__main__':
    main()
