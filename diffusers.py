import torch

class DDPMDiffuser():
    """
        Denoising Diffusion Probabilistic Model (DDPM) diffuser.
        Reference: https://arxiv.org/pdf/2006.11239.pdf

        Parameters
        ----------
    
        betas: torch.Tensor of shape [T]
            Beta values for each time step
    """

    def __init__(self, betas, device='cuda:0'):
        self.device = device
        self.betas = betas.to(device)
        self.alphas = 1 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device)

    def diffuse(self, X_0, t):
        """
            Diffuse data X_0 at time step t.

            Parameters
            ----------
            t: int
                Time step
            X_0: torch.Tensor of shape [N, D]
                Data to be diffused

            Returns
            -------
            X_t: torch.Tensor of shape [N, D]
                Diffused data
            eps: torch.Tensor of shape [N, D]
                Noise added to data
        """

        eps = torch.randn(X_0.size(), dtype=torch.float32, device=self.device)
        X_t = torch.sqrt(self.alpha_bars[t]) * X_0 + torch.sqrt(1 - self.alpha_bars[t]) * eps
        return X_t, eps
