import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split

from dataclasses import dataclass
from tqdm import tqdm

import nn_modules as mynn

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
    train_split : float = 0.9

class MyResNet18(nn.Module):
    def 
    
def train_test_model(config: Config):
    dataset = CIFAR10(
        root=".",
        download=False,
        transform=T.Compose([
            T.ToTensor(),
        ])
    )
    # print(len(dataset))
    train_split = int(config.train_split * len(dataset))
    train, test = random_split(dataset, [train_split, len(dataset) - train_split])
    traindl = DataLoader(train, batch_size=config.batch_size, shuffle=True)
    traindl = DataLoader(test, batch_size=config.batch_size, shuffle=False)

    # model = MyResNet18().to(device)

def test_modules(config : Config):
    b, c_in, w_in, h_in = (4, 3, 32, 32)
    in_tensor = torch.randn(b, c_in, w_in, h_in)
    print(f"in_tensor.shape {in_tensor.shape}")

def main():
    config = Config()
    test_modules(config)

if __name__ == '__main__':
    main()
