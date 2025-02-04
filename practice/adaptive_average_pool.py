import torch
import torch.nn as nn
import math

class MyAdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size):
        """
        Args:
            output_size (int or tuple): The target output size (H_out, W_out). If an integer is provided,
                                        the output will be square.
        """
        super(MyAdaptiveAvgPool2d, self).__init__()
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (N, C, H, W)
            
        Returns:
            Tensor: Output tensor of shape (N, C, H_out, W_out) where each element is the average
                    of a region from the input.
        """
        N, C, H, W = x.shape
        out_H, out_W = self.output_size

        # Create an empty tensor to hold the output.
        output = torch.empty((N, C, out_H, out_W), device=x.device, dtype=x.dtype)

        # Compute the pooling regions and apply average pooling to each region.
        for i in range(out_H):
            # Compute the start and end indices for the height dimension.
            h_start = math.floor(i * H / out_H)
            h_end = math.ceil((i + 1) * H / out_H)
            for j in range(out_W):
                # Compute the start and end indices for the width dimension.
                w_start = math.floor(j * W / out_W)
                w_end = math.ceil((j + 1) * W / out_W)
                # Extract the region from the input.
                region = x[:, :, h_start:h_end, w_start:w_end]
                # Compute the average value over the spatial dimensions (h and w)
                # The mean is computed separately for each sample and channel.
                output[:, :, i, j] = region.mean(dim=(-1, -2))
                
        return output

# Example usage:
if __name__ == '__main__':
    # Create a random tensor with shape (batch_size, channels, height, width)
    x = torch.randn(2, 3, 32, 32)
    
    # Instantiate our custom adaptive average pooling layer to produce a 1x1 output per channel.
    adaptive_pool = MyAdaptiveAvgPool2d((1, 1))
    output = adaptive_pool(x)
    
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

