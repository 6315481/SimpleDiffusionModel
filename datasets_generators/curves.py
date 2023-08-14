import torch

def generate_curves(N: int, num_sprit: int, num_frequency: int, radius: float=10, device='cuda:0'):
    """
    Generates a dataset of curves.

    Parameters
    ----------
    N: int
        Number of curves to generate
    num_sprit: int
        Number of points per curve
    num_frequency: int 
        Number of frequencies to use
    """
    frequencies = torch.linspace(1, num_frequency, steps = num_frequency)
    sprits = 2 * torch.pi * torch.linspace(0, 1, steps = num_sprit).view(-1, 1)
    thetas = sprits * frequencies

    decay_rate = torch.cat([torch.tensor([1.]), torch.ones(num_frequency - 1) / 2], dim=0)
    coefficients = radius * (1 + torch.rand(N, num_frequency)) * torch.cumprod(decay_rate, dim=0)

    x = torch.matmul(coefficients, torch.cos(thetas).transpose(0, 1))
    y = torch.matmul(coefficients, torch.sin(thetas).transpose(0, 1))

    data = torch.stack([x, y], dim=2).transpose(1, 2).to(device)

    return data