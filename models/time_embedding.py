import torch

class SinusoidalTimeEmb1d(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        emb = self.dim / 2 * torch.log(10000)
        emb = torch.exp(torch.arange(self.dim, device=device) * -emb)
        emb = torch.sin(t * emb)

        return emb