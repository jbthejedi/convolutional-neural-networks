import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from dataclasses import dataclass

# Import your custom convolution module.
from convolution2d import MyConv2d

# Set the random seed and device.
seed = 1337
torch.manual_seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class TrainingConfig:
    n_epochs: int = 1
    image_size: int = 32
    p_split: float = 0.8
    batch_size: int = 32


class BasicBlock(nn.Module):
    """
    A simple residual block with two convolutional layers.
    If the block changes the number of channels or the spatial dimensions,
    it creates a shortcut (downsampling) so that the addition works.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # First convolution: may change spatial dimensions if stride > 1.
        self.conv1 = MyConv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second convolution: always stride 1.
        self.conv2 = MyConv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Create a shortcut if the input and output dimensions differ.
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                MyConv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, padding=0, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply the shortcut (downsample) if necessary.
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    """
    A simplified ResNet-18 implementation that uses the BasicBlock.
    The architecture is:
       conv1 --> bn --> relu --> maxpool -->
       layer1 (2 blocks) -->
       layer2 (2 blocks) -->
       layer3 (2 blocks) -->
       layer4 (2 blocks) -->
       adaptive avg pool --> fc
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # Initial convolution, batch normalization, ReLU, and max pooling.
        self.conv1 = MyConv2d(
            in_channels=3, out_channels=64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Create the four residual layers.
        # Each layer is built with a helper method that stacks two blocks.
        self.layer1 = self._make_layer(in_channels=64,  out_channels=64,  num_blocks=2, stride=1)
        self.layer2 = self._make_layer(in_channels=64,  out_channels=128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(in_channels=128, out_channels=256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(in_channels=256, out_channels=512, num_blocks=2, stride=2)

        # Global average pooling and a fully connected layer.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """
        Create a layer (sequence) of residual blocks.
        The first block may downsample (if stride > 1 or if channel numbers differ),
        and the rest use stride=1.
        """
        layers = []
        # First block with the specified stride.
        layers.append(BasicBlock(in_channels, out_channels, stride))
        # Remaining blocks: in_channels == out_channels.
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        # Weight initialization for different modules.
        for m in self.modules():
            if isinstance(m, MyConv2d):
                # You can add custom initialization here if needed.
                pass
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Initial block.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual layers.
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Classification head.
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def train_and_evaluate(config: TrainingConfig):
    # Define data transforms.
    transform = T.Compose([
        T.Resize((config.image_size, config.image_size)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load the CIFAR10 dataset.
    dataset = datasets.CIFAR10(root="data", download=True, transform=transform)
    train_size = int(config.p_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Create the model and move it to the proper device.
    model = ResNet18(num_classes=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Training loop.
    for epoch in range(1, config.n_epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # Use tqdm for a progress bar.
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")


def main():
    config = TrainingConfig(n_epochs=1)
    train_and_evaluate(config)


if __name__ == '__main__':
    main()


