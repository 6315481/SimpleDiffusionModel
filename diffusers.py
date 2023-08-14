import torch

class DDPMDiffuser1d():
    """
        Denoising Diffusion Probabilistic Model (DDPM) diffuser.
        Reference: https://arxiv.org/pdf/2006.11239.pdf

        Parameters
        ----------
    
        betas: torch.Tensor of shape [T]
            Beta values for each time step
    """

    def __init__(self, betas):
        self.device = betas.device
        self.betas = betas
        self.alphas = 1 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(self.device)

    def diffuse(self, X_0, t):
        """
            Diffuse data X_0 at time step t.

            Parameters
            ----------
            t: torch.Tensor of shape [N]
                Time step
            X_0: torch.Tensor of shape [N, C, D]
                Data to be diffused

            Returns
            -------
            X_t: torch.Tensor of shape [N, C, D]
                Diffused data
            eps: torch.Tensor of shape [N, C, D]
                Noise added to data
        """

        eps = torch.randn(X_0.size(), dtype=torch.float32, device=self.device)

        mean_scale = torch.sqrt(self.alpha_bars[t])[:, None, None]
        var_scale = torch.sqrt(1 - self.alpha_bars[t])[:, None, None]
        X_t = mean_scale * X_0 + var_scale * eps
        return X_t, eps

class DDPMDiffuser2d():
    """
        Denoising Diffusion Probabilistic Model (DDPM) diffuser for 2D data.
        Reference: https://arxiv.org/pdf/2006.11239.pdf

        Parameters
        ----------
    
        betas: torch.Tensor of shape [T]
            Beta values for each time step
    """

    def __init__(self, betas):
        self.device = betas.device
        self.betas = betas
        self.alphas = 1 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(self.device)

    def diffuse(self, X_0, t):
        """
            Diffuse 2D data X_0 at time step t.

            Parameters
            ----------
            t: torch.Tensor of shape [N]
                Time step
            X_0: torch.Tensor of shape [N, C, H, W]
                2D Data to be diffused

            Returns
            -------
            X_t: torch.Tensor of shape [N, C, H, W]
                Diffused 2D data
            eps: torch.Tensor of shape [N, C, H, W]
                Noise added to 2D data
        """

        eps = torch.randn(X_0.size(), dtype=torch.float32, device=self.device)

        mean_scale = torch.sqrt(self.alpha_bars[t])[:, None, None, None]
        var_scale = torch.sqrt(1 - self.alpha_bars[t])[:, None, None, None]
        X_t = mean_scale * X_0 + var_scale * eps
        return X_t, eps