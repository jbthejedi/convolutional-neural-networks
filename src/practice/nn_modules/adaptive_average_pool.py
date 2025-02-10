import torch
import torch.nn as nn
import torch.nn.functional as F

class MyAdaptiveAvgPool2dEfficient(nn.Module):
    def __init__(self, output_size):
        """
        Args:
            output_size (int or tuple): The desired output size (H_out, W_out).
                If an integer is provided, the output is assumed to be square.
        """
        super().__init__()
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (N, C, H, W)
            
        Returns:
            Tensor: Output tensor of shape (N, C, H_out, W_out)
        """
        N, C, H, W = x.shape
        out_H, out_W = self.output_size

        # For a fully vectorized implementation using unfold, we assume that H and W
        # are divisible by out_H and out_W, respectively.
        if H % out_H != 0 or W % out_W != 0:
            raise ValueError("For this efficient implementation, H and W must be divisible by out_H and out_W.")

        # Determine the size of each pooling region (kernel) and stride.
        kernel_h = H // out_H
        kernel_w = W // out_W
        kernel_size = (kernel_h, kernel_w)
        stride = kernel_size  # Non-overlapping regions

        # Unfold extracts sliding local blocks from the input.
        # The result has shape: (N, C * kernel_h * kernel_w, out_H*out_W)
        x_unfold = F.unfold(x, kernel_size=kernel_size, stride=stride)
        
        # Reshape to separate the patch (kernel) dimension.
        # New shape: (N, C, kernel_h * kernel_w, out_H, out_W)
        x_unfold = x_unfold.view(N, C, kernel_h * kernel_w, out_H, out_W)
        
        # Compute the average over the patch dimension (kernel_h * kernel_w).
        output = x_unfold.mean(dim=2)
        return output

# Example usage:
if __name__ == '__main__':
    # Create a batch of 2 images, each with 3 channels and size 32x32.
    x = torch.randn(2, 3, 32, 32)
    
    # Define an adaptive average pooling layer that outputs 4x4 feature maps.
    adaptive_pool = MyAdaptiveAvgPool2dEfficient((4, 4))
    output = adaptive_pool(x)
    
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyAdaptiveAvgPool2dEfficient(nn.Module):
    def __init__(self, output_size):
        """
        Args:
            output_size (int or tuple): The desired output size (H_out, W_out).
                If an integer is provided, the output is assumed to be square.
        """
        super().__init__()
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (N, C, H, W)
            
        Returns:
            Tensor: Output tensor of shape (N, C, H_out, W_out)
        """
        N, C, H, W = x.shape
        out_H, out_W = self.output_size

        # For a fully vectorized implementation using unfold, we assume that H and W
        # are divisible by out_H and out_W, respectively.
        if H % out_H != 0 or W % out_W != 0:
            raise ValueError("For this efficient implementation, H and W must be divisible by out_H and out_W.")

        # Determine the size of each pooling region (kernel) and stride.
        kernel_h = H // out_H
        kernel_w = W // out_W
        kernel_size = (kernel_h, kernel_w)
        stride = kernel_size  # Non-overlapping regions

        # Unfold extracts sliding local blocks from the input.
        # The result has shape: (N, C * kernel_h * kernel_w, out_H*out_W)
        x_unfold = F.unfold(x, kernel_size=kernel_size, stride=stride)
        
        # Reshape to separate the patch (kernel) dimension.
        # New shape: (N, C, kernel_h * kernel_w, out_H, out_W)
        x_unfold = x_unfold.view(N, C, kernel_h * kernel_w, out_H, out_W)
        
        # Compute the average over the patch dimension (kernel_h * kernel_w).
        output = x_unfold.mean(dim=2)
        return output

# Example usage:
if __name__ == '__main__':
    # Create a batch of 2 images, each with 3 channels and size 32x32.
    x = torch.randn(2, 3, 32, 32)
    
    # Define an adaptive average pooling layer that outputs 4x4 feature maps.
    adaptive_pool = MyAdaptiveAvgPool2dEfficient((4, 4))
    output = adaptive_pool(x)
    
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyAdaptiveAvgPool2dEfficient(nn.Module):
    def __init__(self, output_size):
        """
        Args:
            output_size (int or tuple): The desired output size (H_out, W_out).
                If an integer is provided, the output is assumed to be square.
        """
        super().__init__()
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (N, C, H, W)
            
        Returns:
            Tensor: Output tensor of shape (N, C, H_out, W_out)
        """
        N, C, H, W = x.shape
        out_H, out_W = self.output_size

        # For a fully vectorized implementation using unfold, we assume that H and W
        # are divisible by out_H and out_W, respectively.
        if H % out_H != 0 or W % out_W != 0:
            raise ValueError("For this efficient implementation, H and W must be divisible by out_H and out_W.")

        # Determine the size of each pooling region (kernel) and stride.
        kernel_h = H // out_H
        kernel_w = W // out_W
        kernel_size = (kernel_h, kernel_w)
        stride = kernel_size  # Non-overlapping regions

        # Unfold extracts sliding local blocks from the input.
        # The result has shape: (N, C * kernel_h * kernel_w, out_H*out_W)
        x_unfold = F.unfold(x, kernel_size=kernel_size, stride=stride)
        
        # Reshape to separate the patch (kernel) dimension.
        # New shape: (N, C, kernel_h * kernel_w, out_H, out_W)
        x_unfold = x_unfold.view(N, C, kernel_h * kernel_w, out_H, out_W)
        
        # Compute the average over the patch dimension (kernel_h * kernel_w).
        output = x_unfold.mean(dim=2)
        return output

# Example usage:
if __name__ == '__main__':
    # Create a batch of 2 images, each with 3 channels and size 32x32.
    x = torch.randn(2, 3, 32, 32)
    
    # Define an adaptive average pooling layer that outputs 4x4 feature maps.
    adaptive_pool = MyAdaptiveAvgPool2dEfficient((4, 4))
    output = adaptive_pool(x)
    
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyAdaptiveAvgPool2dEfficient(nn.Module):
    def __init__(self, output_size):
        """
        Args:
            output_size (int or tuple): The desired output size (H_out, W_out).
                If an integer is provided, the output is assumed to be square.
        """
        super().__init__()
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (N, C, H, W)
            
        Returns:
            Tensor: Output tensor of shape (N, C, H_out, W_out)
        """
        N, C, H, W = x.shape
        out_H, out_W = self.output_size

        # For a fully vectorized implementation using unfold, we assume that H and W
        # are divisible by out_H and out_W, respectively.
        if H % out_H != 0 or W % out_W != 0:
            raise ValueError("For this efficient implementation, H and W must be divisible by out_H and out_W.")

        # Determine the size of each pooling region (kernel) and stride.
        kernel_h = H // out_H
        kernel_w = W // out_W
        kernel_size = (kernel_h, kernel_w)
        stride = kernel_size  # Non-overlapping regions

        # Unfold extracts sliding local blocks from the input.
        # The result has shape: (N, C * kernel_h * kernel_w, out_H*out_W)
        x_unfold = F.unfold(x, kernel_size=kernel_size, stride=stride)
        
        # Reshape to separate the patch (kernel) dimension.
        # New shape: (N, C, kernel_h * kernel_w, out_H, out_W)
        x_unfold = x_unfold.view(N, C, kernel_h * kernel_w, out_H, out_W)
        
        # Compute the average over the patch dimension (kernel_h * kernel_w).
        output = x_unfold.mean(dim=2)
        return output

# Example usage:
if __name__ == '__main__':
    # Create a batch of 2 images, each with 3 channels and size 32x32.
    x = torch.randn(2, 3, 32, 32)
    
    # Define an adaptive average pooling layer that outputs 4x4 feature maps.
    adaptive_pool = MyAdaptiveAvgPool2dEfficient((4, 4))
    output = adaptive_pool(x)
    
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyAdaptiveAvgPool2dEfficient(nn.Module):
    def __init__(self, output_size):
        """
        Args:
            output_size (int or tuple): The desired output size (H_out, W_out).
                If an integer is provided, the output is assumed to be square.
        """
        super().__init__()
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (N, C, H, W)
            
        Returns:
            Tensor: Output tensor of shape (N, C, H_out, W_out)
        """
        N, C, H, W = x.shape
        out_H, out_W = self.output_size

        # For a fully vectorized implementation using unfold, we assume that H and W
        # are divisible by out_H and out_W, respectively.
        if H % out_H != 0 or W % out_W != 0:
            raise ValueError("For this efficient implementation, H and W must be divisible by out_H and out_W.")

        # Determine the size of each pooling region (kernel) and stride.
        kernel_h = H // out_H
        kernel_w = W // out_W
        kernel_size = (kernel_h, kernel_w)
        stride = kernel_size  # Non-overlapping regions

        # Unfold extracts sliding local blocks from the input.
        # The result has shape: (N, C * kernel_h * kernel_w, out_H*out_W)
        x_unfold = F.unfold(x, kernel_size=kernel_size, stride=stride)
        
        # Reshape to separate the patch (kernel) dimension.
        # New shape: (N, C, kernel_h * kernel_w, out_H, out_W)
        x_unfold = x_unfold.view(N, C, kernel_h * kernel_w, out_H, out_W)
        
        # Compute the average over the patch dimension (kernel_h * kernel_w).
        output = x_unfold.mean(dim=2)
        return output

# Example usage:
if __name__ == '__main__':
    # Create a batch of 2 images, each with 3 channels and size 32x32.
    x = torch.randn(2, 3, 32, 32)
    
    # Define an adaptive average pooling layer that outputs 4x4 feature maps.
    adaptive_pool = MyAdaptiveAvgPool2dEfficient((4, 4))
    output = adaptive_pool(x)
    
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyAdaptiveAvgPool2dEfficient(nn.Module):
    def __init__(self, output_size):
        """
        Args:
            output_size (int or tuple): The desired output size (H_out, W_out).
                If an integer is provided, the output is assumed to be square.
        """
        super().__init__()
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (N, C, H, W)
            
        Returns:
            Tensor: Output tensor of shape (N, C, H_out, W_out)
        """
        N, C, H, W = x.shape
        out_H, out_W = self.output_size

        # For a fully vectorized implementation using unfold, we assume that H and W
        # are divisible by out_H and out_W, respectively.
        if H % out_H != 0 or W % out_W != 0:
            raise ValueError("For this efficient implementation, H and W must be divisible by out_H and out_W.")

        # Determine the size of each pooling region (kernel) and stride.
        kernel_h = H // out_H
        kernel_w = W // out_W
        kernel_size = (kernel_h, kernel_w)
        stride = kernel_size  # Non-overlapping regions

        # Unfold extracts sliding local blocks from the input.
        # The result has shape: (N, C * kernel_h * kernel_w, out_H*out_W)
        x_unfold = F.unfold(x, kernel_size=kernel_size, stride=stride)
        
        # Reshape to separate the patch (kernel) dimension.
        # New shape: (N, C, kernel_h * kernel_w, out_H, out_W)
        x_unfold = x_unfold.view(N, C, kernel_h * kernel_w, out_H, out_W)
        
        # Compute the average over the patch dimension (kernel_h * kernel_w).
        output = x_unfold.mean(dim=2)
        return output

# Example usage:
if __name__ == '__main__':
    # Create a batch of 2 images, each with 3 channels and size 32x32.
    x = torch.randn(2, 3, 32, 32)
    
    # Define an adaptive average pooling layer that outputs 4x4 feature maps.
    adaptive_pool = MyAdaptiveAvgPool2dEfficient((4, 4))
    output = adaptive_pool(x)
    
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyAdaptiveAvgPool2dEfficient(nn.Module):
    def __init__(self, output_size):
        """
        Args:
            output_size (int or tuple): The desired output size (H_out, W_out).
                If an integer is provided, the output is assumed to be square.
        """
        super().__init__()
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (N, C, H, W)
            
        Returns:
            Tensor: Output tensor of shape (N, C, H_out, W_out)
        """
        N, C, H, W = x.shape
        out_H, out_W = self.output_size

        # For a fully vectorized implementation using unfold, we assume that H and W
        # are divisible by out_H and out_W, respectively.
        if H % out_H != 0 or W % out_W != 0:
            raise ValueError("For this efficient implementation, H and W must be divisible by out_H and out_W.")

        # Determine the size of each pooling region (kernel) and stride.
        kernel_h = H // out_H
        kernel_w = W // out_W
        kernel_size = (kernel_h, kernel_w)
        stride = kernel_size  # Non-overlapping regions

        # Unfold extracts sliding local blocks from the input.
        # The result has shape: (N, C * kernel_h * kernel_w, out_H*out_W)
        x_unfold = F.unfold(x, kernel_size=kernel_size, stride=stride)
        
        # Reshape to separate the patch (kernel) dimension.
        # New shape: (N, C, kernel_h * kernel_w, out_H, out_W)
        x_unfold = x_unfold.view(N, C, kernel_h * kernel_w, out_H, out_W)
        
        # Compute the average over the patch dimension (kernel_h * kernel_w).
        output = x_unfold.mean(dim=2)
        return output

# Example usage:
if __name__ == '__main__':
    # Create a batch of 2 images, each with 3 channels and size 32x32.
    x = torch.randn(2, 3, 32, 32)
    
    # Define an adaptive average pooling layer that outputs 4x4 feature maps.
    adaptive_pool = MyAdaptiveAvgPool2dEfficient((4, 4))
    output = adaptive_pool(x)
    
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyAdaptiveAvgPool2dEfficient(nn.Module):
    def __init__(self, output_size):
        """
        Args:
            output_size (int or tuple): The desired output size (H_out, W_out).
                If an integer is provided, the output is assumed to be square.
        """
        super().__init__()
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (N, C, H, W)
            
        Returns:
            Tensor: Output tensor of shape (N, C, H_out, W_out)
        """
        N, C, H, W = x.shape
        out_H, out_W = self.output_size

        # For a fully vectorized implementation using unfold, we assume that H and W
        # are divisible by out_H and out_W, respectively.
        if H % out_H != 0 or W % out_W != 0:
            raise ValueError("For this efficient implementation, H and W must be divisible by out_H and out_W.")

        # Determine the size of each pooling region (kernel) and stride.
        kernel_h = H // out_H
        kernel_w = W // out_W
        kernel_size = (kernel_h, kernel_w)
        stride = kernel_size  # Non-overlapping regions

        # Unfold extracts sliding local blocks from the input.
        # The result has shape: (N, C * kernel_h * kernel_w, out_H*out_W)
        x_unfold = F.unfold(x, kernel_size=kernel_size, stride=stride)
        
        # Reshape to separate the patch (kernel) dimension.
        # New shape: (N, C, kernel_h * kernel_w, out_H, out_W)
        x_unfold = x_unfold.view(N, C, kernel_h * kernel_w, out_H, out_W)
        
        # Compute the average over the patch dimension (kernel_h * kernel_w).
        output = x_unfold.mean(dim=2)
        return output

# Example usage:
if __name__ == '__main__':
    # Create a batch of 2 images, each with 3 channels and size 32x32.
    x = torch.randn(2, 3, 32, 32)
    
    # Define an adaptive average pooling layer that outputs 4x4 feature maps.
    adaptive_pool = MyAdaptiveAvgPool2dEfficient((4, 4))
    output = adaptive_pool(x)
    
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyAdaptiveAvgPool2dEfficient(nn.Module):
    def __init__(self, output_size):
        """
        Args:
            output_size (int or tuple): The desired output size (H_out, W_out).
                If an integer is provided, the output is assumed to be square.
        """
        super().__init__()
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (N, C, H, W)
            
        Returns:
            Tensor: Output tensor of shape (N, C, H_out, W_out)
        """
        N, C, H, W = x.shape
        out_H, out_W = self.output_size

        # For a fully vectorized implementation using unfold, we assume that H and W
        # are divisible by out_H and out_W, respectively.
        if H % out_H != 0 or W % out_W != 0:
            raise ValueError("For this efficient implementation, H and W must be divisible by out_H and out_W.")

        # Determine the size of each pooling region (kernel) and stride.
        kernel_h = H // out_H
        kernel_w = W // out_W
        kernel_size = (kernel_h, kernel_w)
        stride = kernel_size  # Non-overlapping regions

        # Unfold extracts sliding local blocks from the input.
        # The result has shape: (N, C * kernel_h * kernel_w, out_H*out_W)
        x_unfold = F.unfold(x, kernel_size=kernel_size, stride=stride)
        
        # Reshape to separate the patch (kernel) dimension.
        # New shape: (N, C, kernel_h * kernel_w, out_H, out_W)
        x_unfold = x_unfold.view(N, C, kernel_h * kernel_w, out_H, out_W)
        
        # Compute the average over the patch dimension (kernel_h * kernel_w).
        output = x_unfold.mean(dim=2)
        return output

# Example usage:
if __name__ == '__main__':
    # Create a batch of 2 images, each with 3 channels and size 32x32.
    x = torch.randn(2, 3, 32, 32)
    
    # Define an adaptive average pooling layer that outputs 4x4 feature maps.
    adaptive_pool = MyAdaptiveAvgPool2dEfficient((4, 4))
    output = adaptive_pool(x)
    
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

