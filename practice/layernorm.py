
class LayerNorm:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def __call__(self, x):
        xmean = x.mean(-1, keepdim=True)
        xvar = x.var(-1, keepdim=True)
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        # print(f"ln.shape {self.out.shape}")
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

