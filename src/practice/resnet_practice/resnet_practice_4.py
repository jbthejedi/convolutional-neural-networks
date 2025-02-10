import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from dataclasses import dataclass
# from convolution2d import MyConv2d

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class TrainingConfig:
    batch_size : int = 32
    n_epochs : int = 1
    image_size : int = 32
    normalize_shape : tuple = ((0.5,), (0.5,), (0.5,))
    train_split : float = 0.8

# IOKSP
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, padding=0, bias=False
                ),
                nn.BatchNorm2d(out_channels)
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

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class MyResNet18(nn.Module):
    def __init__(self, num_classes=10):
        """
        conv1 -> bn1 -> relu -> maxpool
        layer1 (2blocks) ->
        layer2 (2blocks) ->
        layer3 (2blocks) ->
        layer4 (2blocks) ->
        avgpool -> fc
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64,
            kernel_size=7, stride=2, padding=3
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            in_channels=64, out_channels=64, n_blocks=2, stride=1
        )
        self.layer2 = self._make_layer(
            in_channels=64, out_channels=128, n_blocks=2, stride=2
        )
        self.layer3 = self._make_layer(
            in_channels=128, out_channels=256, n_blocks=2, stride=2
        )
        self.layer4 = self._make_layer(
            in_channels=256, out_channels=512, n_blocks=2, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, n_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # pass
                # # Use Kaiming Normal initialization
                # # for Conv2d layers.
                # nn.init.kaiming_normal_(
                #     m.weight, mode='fan_out', nonlinearity='relu'
                # )
                # # If the layer has a bias term
                # # (it might be None if bias=False), initialize it to zero.
                # if m.bias is not None:
                #     nn.init.zeros_(m.bias)
                nn.init.normal_(m.weight, 0, 0.01)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        pass
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
        out = self.fc(x)

        return out

def visualize_predictions(model, dataloader, num_images=0):
    model.eval()
    images_shown = 0
    plt.figure(figsize=(15, 5))
    for inputs, labels in dataloader:
        logits = model(inputs)
        _, preds = torch.max(logits, 1)

        for i in range(inputs.size(0)): # iterate over batch dimension
            if images_shown >= num_images:
                break
            img = inputs[i].cpu().squeeze()

            true_label = labels[i].item()
            pred_label = preds[i].item()

            plt.subplot(2, num_images//2, images_shown+1)
            plt.imshow(img, cmap='gray')
            plt.title(f"True {true_label} Pred {pred_label}")
            plt.axis('off')

            images_shown += 1
        if images_shown >= num_images:
            break
    plt.tight_layout()
    plt.show()

def train_test_model(config : TrainingConfig):
    dataset = CIFAR10(
        root="../../vision-transformer",
        transform=T.Compose([
            T.Resize((config.image_size, config.image_size)),
            T.ToTensor(),
        ])
    )
    train_len = int(config.train_split * len(dataset))
    train, test = random_split(dataset, [train_len, len(dataset) - train_len])
    traindl = DataLoader(train, batch_size=config.batch_size, shuffle=True)
    testdl = DataLoader(test, batch_size=config.batch_size, shuffle=False)

    model = MyResNet18(num_classes=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    
    for epoch in range(1, config.n_epochs + 1):
        tqdm.write(f"Epoch {epoch}/{config.n_epochs}")
        
        # Training phase
        model.train()
        total_train_loss = 0.0
        train_correct = 0
        total_train = 0

        with tqdm(traindl, desc="Training", unit="batch", leave=False) as pbar:
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

        with tqdm(testdl, desc="Validation", unit="batch", leave=False) as pbar:
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
    visualize_predictions(model, test_dl, num_images=10)

def main():
    config = TrainingConfig()
    train_test_model(config)

if __name__ == '__main__':
    main()
