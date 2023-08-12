import torch

class DDPMSampler():
    """
        Denoising Diffusion Probabilistic Model (DDPM) sampler.
        Reference: https://arxiv.org/pdf/2006.11239.pdf

        Parameters
        ----------
        betas: torch.Tensor of shape [T]
            Beta values for each time step
    """

    def __init__(self, betas: torch.Tensor, noise_model, device: str = 'cuda:0'):
        self.betas = betas.to(device)
        self.alphas = 1 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device)
        self.noise_model = noise_model
        self.device = device

    def sample(self, x, t):
        """
            Sample from the DDPM model.

            Parameters
            ----------
            x: torch.tensor
                Sample of previous time step

            t: int
                Time step
                
            Returns
            -------
            samples: torch.Tensor
                Sample at t - 1 from the DDPM model
        """

        scale = (1 - self.alphas[t+1]) / torch.sqrt(1 - self.alpha_bars[t+1])
        sigma =  (1 - self.alpha_bars[t]) * (1 - self.alphas[t+1]) / (1 - self.alpha_bars[t+1])
        x = (1 / torch.sqrt(self.alphas[t+1])) * (x - scale * self.noise_model(x, t+1)) + torch.sqrt(sigma) * torch.randn_like(x).to(self.device)
        return x
