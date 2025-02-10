import torch
import torch.nn as nn

class MyBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        """
        Args:
            num_features (int): Number of channels in the input.
            eps (float): A small value added to the denominator for numerical stability.
            momentum (float): The momentum factor for updating the running statistics.
            affine (bool): If True, includes learnable scale (gamma) and shift (beta) parameters.
            track_running_stats (bool): If True, keeps track of running mean and variance.
        """
        super(MyBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            # Learnable scale and shift parameters (gamma and beta)
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

        if self.track_running_stats:
            # Running statistics are not learnable parameters, so we register them as buffers.
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.running_mean = None
            self.running_var = None

    def forward(self, x):
        """
        Applies batch normalization over a 4D input (N, C, H, W).

        During training, computes the mean and variance of each channel using the current batch,
        updates the running statistics, and normalizes the batch.
        During evaluation, uses the stored running statistics.
        """
        if self.training:
            # Compute mean and variance across (N, H, W) for each channel
            batch_mean = x.mean(dim=[0, 2, 3])
            # Use unbiased=False (i.e. divide by N rather than N-1) similar to PyTorch's default
            batch_var = x.var(dim=[0, 2, 3], unbiased=False)
            
            if self.track_running_stats:
                # Update running mean and variance
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            # Normalize using batch statistics
            mean = batch_mean
            var = batch_var
        else:
            # During evaluation, use the stored running statistics
            mean = self.running_mean
            var = self.running_var

        # Normalize the input: subtract mean and divide by sqrt(var + eps)
        # The unsqueeze calls ensure proper broadcasting across (N, C, H, W)
        x_norm = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        
        # If affine is True, apply the learnable scale (gamma) and shift (beta)
        if self.affine:
            x_norm = self.gamma[None, :, None, None] * x_norm + self.beta[None, :, None, None]
            
        return x_norm

# Example usage:
if __name__ == '__main__':
    # Create a batch of 8 images, each with 3 channels and size 32x32.
    x = torch.randn(8, 3, 32, 32)
    
    # Instantiate our custom batch normalization layer.
    bn_layer = MyBatchNorm2d(num_features=3)
    
    # Training mode: statistics will be computed from the current batch.
    bn_layer.train()
    out_train = bn_layer(x)
    print("Output during training:", out_train.shape)
    
    # Evaluation mode: running statistics are used.
    bn_layer.eval()
    out_eval = bn_layer(x)
    print("Output during evaluation:", out_eval.shape)

