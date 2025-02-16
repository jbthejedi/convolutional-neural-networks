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
    normalize_shape : tuple = (0.5, 0.5, 0.5)

class BasicBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride=1
    ):
        """
        x      ->
        identity = x
        conv   ->
        bn     ->
        relu   ->
        
        conv   ->
        bn     ->

        if downsample then downsample(identity)
        x = x + identity
        return x
        """
        super().__init__()
        self.conv1 = mynn.MyConv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = mynn.MyBatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = mynn.MyConv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1
        )
        self.bn2 = mynn.MyBatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                mynn.MyConv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, padding=0
                ),
                mynn.MyBatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)
        x = x + identity

        return x

class MyResNet18(nn.Module):
    def __init__(self, num_classes=10):
        """
        conv    ->
        bn      ->
        relu    ->
        maxpool ->

        block 1 ->
        block 2 ->
        block 3 ->
        block 4 ->

        avg pool ->
        fc       -> out
        """
        super().__init__()
        self.conv1 = mynn.MyConv2d(
            in_channels=3, out_channels=64,
            kernel_size=7, stride=2, padding=3
        )
        self.bn1 = mynn.MyBatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = mynn.MyMaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, n_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, n_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, n_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, n_blocks=2, stride=2)

        self.avgpool = mynn.MyAvgPool2d((1, 1)) # (B, 512, h_out, w_out)
        self.fc = nn.Linear(512, num_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x) # -> (B, 512, 1, 1)
        x = torch.flatten(x, 1) # -> (B, 512)
        x = self.fc(x)
        return x

    def _make_layer(self, in_channels, out_channels, n_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self, x):
        for m in self.modules():
            if isinstance(m, nn.MyConv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01) 
            if isinstance(m, nn.MyBatchNorm2d):
                nn.init.ones_(m.gamma)
                nn.init.zeros_(m.beta)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.normal_(m.bias)

    
def train_test_model(config: Config):
    dataset = CIFAR10(
        root="../../../vision-transformer/data",
        # root="./data",
        download=False,
        transform=T.Compose([
            T.ToTensor(),
            T.Normalize(config.normalize_shape, config.normalize_shape),
        ])
    )
    import random
    from torch.utils.data import Subset
    indices = random.sample(range(len(dataset)), 1000)
    dataset = Subset(dataset, indices)

    train_split = int(config.train_split * len(dataset))
    train, test = random_split(dataset, [train_split, len(dataset) - train_split])
    traindl = DataLoader(train, batch_size=config.batch_size, shuffle=True)
    testdl = DataLoader(test, batch_size=config.batch_size, shuffle=False)

    model = MyResNet18(num_classes=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.n_epochs):
        tqdm.write(f"Epoch {epoch}/{config.n_epochs}")

        model.train()
        with tqdm(traindl, desc="Training") as pbar:
            train_total_correct = 0
            train_total_loss = 0.0
            train_total_size = 0
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, dim=1)
                train_total_correct += (== labels).sum().item()
                train_total_loss += loss.item()
                train_total_size += labels.size(0)

            train_epoch_loss = train_total_loss / train_total_size
            train_epoch_acc = train_total_correct / train_total_size

        model.eval()
        with tqdm(testdl, desc="Training") as pbar:
            val_total_correct = 0
            val_total_loss = 0.0
            val_total_size = 0
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, dim=1)

                val_total_correct += (preds == labels).sum().item()
                val_total_loss += loss.item()
                val_total_size += labels.size(0)

            val_epoch_loss = val_total_loss / val_total_size
            val_epoch_acc = val_total_correct / val_total_size

        pbar.set_postfix(
            Train_Loss=f"{train_epoch_loss}:.4f",
            Val_Loss=f"{val_epoch_loss}:.4f",
            Train_Acc=f"{train_epoch_acc}:2f",
            Val_Acc=f"{val_epoch_acc}:.2f"
        )


def test_modules(config : Config):
    b, c_in, w_in, h_in = (4, 3, 32, 32)
    in_tensor = torch.randn(b, c_in, w_in, h_in)
    print(f"in_tensor.shape {in_tensor.shape}")

def main():
    config = Config()
    test_modules(config)
    train_test_model(config)

if __name__ == '__main__':
    main()










