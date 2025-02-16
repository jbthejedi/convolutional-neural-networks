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
    train_split : float = 0.9
    normalize_shape : tuple = (0.5, 0.5, 0.5)

class MyConv2d(nn.Module):
    def __init__(self,
        in_channels, out_channels,
        kernel_size, stride, padding
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01)

    def forward(self, x):
        """
        h_out = (h_in + 2*p - k) // s + 1
        w_out = (w_in + 2*p - k) // s + 1

        out.shape = (b, c_out, h_out, w_out)
        """
        b, c_in, h_in, w_in = x.shape
        k, s, p = self.kernel_size, self.stride, self.padding
        x_unfold = F.unfold( # -> (b, c_in*k*k, L), where L = h_out * w_out
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        x_unfold_t = x_unfold.transpose(1, 2) # (b, L, c_in*k*k)
        weights = self.weights.view(self.out_channels, self.in_channels*k*k) # (c_out, c_in*k*k)
        out = x_unfold_t @ weights.t() # (b, L, c_out)
        out = out.transpose(1, 2)
        h_out = (h_in + 2 * p - k) // s + 1
        w_out = (w_in + 2 * p - k) // s + 1
        out = out.view(b, -1, h_out, w_out)
        return out

class MyMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride      = stride
        self.padding     = padding

    def forward(self, x):
        b, c_in, h_in, w_in = x.shape
        k, s, p = self.kernel_size, self.stride, self.padding

        # x_unfold.shape => (b, c_in*k*k, L), where L = h_out * w_out
        x_unfold = F.unfold(x, kernel_size=k, stride=s, padding=p)
        x_unfold = x_unfold.view(b, c_in, k*k, -1)
        x_max, _ = x_unfold.max(dim=2)
        h_out = (h_in + 2*p - k) // s + 1
        w_out = (w_in + 2*p - k) // s + 1
        out = x_max.view(b, c_in, h_out, w_out)

        return out

class MyAvgPool2d(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        b, c_in, h_in, w_in = x.shape
        h_out, w_out = self.output_size
        kernel_h = h_in // h_out
        kernel_w = w_in // w_out
        kernel = (kernel_h, kernel_w)
        x_unfold = F.unfold( # (b, c_in*k*k, L), L = 1*1 = 1
            x,
            kernel_size=kernel,
            stride=kernel,
        )
        x_unfold = x_unfold.view(b, c_in, kernel_h*kernel_w, h_out, w_out)
        out = x_unfold.mean(dim=2)
        return out

class MyLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum

        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x_mean = x.mean(-1, keepdim=True)
        x_var = x.var(-1, keepdim=True)
        x_hat = (x - x_mean) / torch.sqrt(x_var + self.eps)
        x_hat = x_hat * self.gamma + self.beta

        return x_hat

class MyBatchNorm2d(nn.Module):
    def __init__(
        self,
        num_features, # num_channels/num_feature_maps
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        # Affine = use learnable scale (gamma) and shift (beta)
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.running_mean = None
            self.running_var = None

    def forward(self, x):
        # If the model is training, than we calculate 
        # the current mean and var using the past mean
        # and var with momentum.
        if self.training:
            # x.shape = (b, c_in, h_in, w_in)
            # Calculate fe feature map the mean/var across all images in batch.
            batch_mean = x.mean(dim=[0, 2, 3])
            batch_var = x.var(dim=[0, 2, 3], unbiased=False)

            if self.track_running_stats:
                # Momentum => running_mean is more important than batch_mean
                # if mom = 0.1 => 1 - 0.1 = .9 and .9*running_mean + .1*batch_mean 
                # puts more importance on running_mean
                self.running_mean = (
                    (1 - self.momentum)*self.running_mean + self.momentum*batch_mean
                )
                self.running_var = (
                    (1 - self.momentum)*self.running_var + self.momentum*batch_var
                )
            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # x_hat = x - mu / sqrt(sigma)
        x_norm = (
            (x - mean[None, :, None, None])
            / torch.sqrt(var[None, :, None, None] + self.eps)
        )
        if self.affine:
            x_norm = (
                (self.gamma[None, :, None, None] * x_norm)
                + self.beta[None, :, None, None]
            )

        return x


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
        self.conv1 = MyConv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = MyBatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = MyConv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1
        )
        self.bn2 = MyBatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                MyConv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, padding=0
                ),
                MyBatchNorm2d(out_channels)
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
        self.conv1 = MyConv2d(
            in_channels=3, out_channels=64,
            kernel_size=7, stride=2, padding=3
        )
        self.bn1 = MyBatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = MyMaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, n_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, n_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, n_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, n_blocks=2, stride=2)

        self.avgpool = MyAvgPool2d((1, 1)) # (B, 512, h_out, w_out)
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
    # import random
    # indices = random.sample(range(len(dataset)), 1000)
    # dataset = Subset(dataset, indices)

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
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                correct = (preds == labels).sum().item()
                
                train_total_correct += correct
                train_total_loss += loss.item()
                train_total_size += labels.size(0)
                pbar.set_postfix(loss=loss.item())

            train_epoch_loss = train_total_loss / train_total_size
            train_epoch_acc = train_total_correct / train_total_size

        model.eval()
        with tqdm(testdl, desc="Training") as pbar:
            val_correct = 0
            val_total_loss = 0.0
            val_total_size = 0
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.no_grad():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_total_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total_size += labels.size(0)
                pbar.set_postfix(loss=loss.item())

            val_epoch_loss = val_total_loss / val_total_size
            val_epoch_acc = val_correct / val_total_size

            Train_Loss=f"{train_epoch_loss}:.4f",
            Val_Loss=f"{val_epoch_loss}:.4f",
            Train_Acc=f"{train_epoch_acc}:2f",
            Val_Acc=f"{val_epoch_acc}:.2f"


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











