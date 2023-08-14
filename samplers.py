import torch

class DDPMSampler1d():
    """
        Denoising Diffusion Probabilistic Model (DDPM) sampler.
        Reference: https://arxiv.org/pdf/2006.11239.pdf

        Parameters
        ----------
        betas: torch.Tensor of shape [T]
            Beta values for each time step
    """

    def __init__(self, betas: torch.Tensor, noise_model):
        self.device = betas.device
        self.betas = betas
        self.alphas = 1 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.noise_model = noise_model

    def sample(self, x, t):
        """
            Sample from the DDPM model.

            Parameters
            ----------
            x: torch.tensor of shape [N, C, D]
                Sample of previous time step

            t: torch.tensor of shape [N]
                Time step
                
            Returns
            -------
            samples: torch.Tensor
                Sample at t - 1 from the DDPM model
        """

        scale = (1 - self.alphas[t+1]) / torch.sqrt(1 - self.alpha_bars[t+1])
        sigma =  (1 - self.alpha_bars[t]) * (1 - self.alphas[t+1]) / (1 - self.alpha_bars[t+1])
        inv = (1 / torch.sqrt(self.alphas[t+1]))

        scale = scale[:, None, None]
        sigma = sigma[:, None, None]
        inv = inv[:, None, None]

        sample = inv * (x - scale * self.noise_model(x, t+1)) + torch.sqrt(sigma) * torch.randn_like(x).to(self.device)
        return sample

class DDPMSampler2d():
    """
        Denoising Diffusion Probabilistic Model (DDPM) sampler for 2D data.
        Reference: https://arxiv.org/pdf/2006.11239.pdf

        Parameters
        ----------
        betas: torch.Tensor of shape [T]
            Beta values for each time step
    """

    def __init__(self, betas: torch.Tensor, noise_model):
        self.device = betas.device
        self.betas = betas
        self.alphas = 1 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.noise_model = noise_model

    def sample(self, x, t):
        """
            Sample from the DDPM model for 2D data.

            Parameters
            ----------
            x: torch.tensor of shape [N, C, H, W]
                2D Sample of previous time step

            t: torch.tensor of shape [N]
                Time step
                
            Returns
            -------
            samples: torch.Tensor of shape [N, C, H, W]
                2D Sample at t - 1 from the DDPM model
        """

        scale = (1 - self.alphas[t+1]) / torch.sqrt(1 - self.alpha_bars[t+1])
        sigma =  (1 - self.alpha_bars[t]) * (1 - self.alphas[t+1]) / (1 - self.alpha_bars[t+1])
        inv = (1 / torch.sqrt(self.alphas[t+1]))
        
        scale = scale[:, None, None, None]
        sigma = sigma[:, None, None, None]
        inv = inv[:, None, None, None]

        sample = inv * (x - scale * self.noise_model(x, t+1)) + torch.sqrt(sigma) * torch.randn_like(x).to(self.device)
        return sample