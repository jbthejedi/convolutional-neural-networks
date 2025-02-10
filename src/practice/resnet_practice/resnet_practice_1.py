import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from convolution2d import MyConv2d
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from dataclasses import dataclass

# Set seed and device
seed = 1337
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(seed)
print(f"Using device: {device}")
print(f"Seed: {seed}")

@dataclass
class TrainingConfig:
    num_classes: int = 10
    num_epochs: int = 10
    train_split_ratio: float = 0.8
    image_size: int = 32
    batch_size: int = 32

class BasicBlock(nn.Module):
    expansion = 1  # For BasicBlock, output channels == out_channels

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        # First convolutional layer
        self.conv1 = MyConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second convolutional layer
        self.conv2 = MyConv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  # To adjust dimensions when needed

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class MyResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        """
        Args:
            block:       The residual block class (BasicBlock or Bottleneck)
            layers:      List with the number of blocks in each layer (e.g. [2,2,2,2] for ResNet-18)
            num_classes: Number of output classes
        """
        super().__init__()
        self.current_channels = 64  # Initial number of channels

        # Initial convolution: 7x7, stride=2, padding=3
        self.conv1 = MyConv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Build residual layers
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Global average pool and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, MyConv2d):
                # Custom initialization if needed
                pass
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.current_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                MyConv2d(
                    in_channels=self.current_channels,
                    out_channels=out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.current_channels, out_channels, stride, downsample))
        self.current_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.current_channels, out_channels))

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

def train_and_evaluate(training_config: TrainingConfig):
    transform = transforms.Compose([
        transforms.Resize((training_config.image_size, training_config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(training_config.image_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    data_dir = "./data"
    dataset = CIFAR10(root=data_dir, download=True, transform=transform)

    train_size = int(training_config.train_split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=training_config.batch_size, shuffle=False)

    model = MyResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=training_config.num_classes)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, training_config.num_epochs + 1):
        tqdm.write(f"Epoch {epoch}/{training_config.num_epochs}")
        
        # Training phase
        model.train()
        total_train_loss = 0.0
        train_correct = 0
        total_train = 0

        with tqdm(train_loader, desc="Training", unit="batch", leave=False) as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                train_correct += (preds == labels).sum().item()
                total_train += labels.size(0)
                pbar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / total_train
        train_accuracy = train_correct / total_train

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        val_correct = 0
        total_val = 0

        with tqdm(test_loader, desc="Validation", unit="batch", leave=False) as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                with torch.no_grad():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                total_val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                total_val += labels.size(0)
                pbar.set_postfix(loss=loss.item())

        avg_val_loss = total_val_loss / total_val
        val_accuracy = val_correct / total_val

        tqdm.write(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}")
        tqdm.write(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}\n")

def main():
    training_config = TrainingConfig()  # Use default training settings
    train_and_evaluate(training_config)

if __name__ == '__main__':
    main()

