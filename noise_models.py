import torch

class NoiseModel1Rank():
    def __init__(self, model, n_time_steps: int, device='cuda:0'):
         self.model = model
         self.n_time_steps = n_time_steps
         self.device = device

    def __call__(self, X: torch.tensor, t: int):
        times = t * torch.ones(X.shape[0]).view(X.shape[0], 1).to(self.device) / self.n_time_steps
        combined = torch.cat([X, times], dim=1).to(self.device)
        return self.model(combined)