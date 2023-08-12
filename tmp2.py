import torch
from noise_models import NoiseModel1Rank
from samplers import DDPMSampler
from utils import plot_pdf, plot_paths

T = 1000
model = torch.load("model_weight.pth")
noise_model = NoiseModel1Rank(model, T, device='cuda:0')
sampler = DDPMSampler(torch.linspace(10e-4, 0.02, T).cuda(), noise_model, device='cuda:0')

with torch.no_grad():
    n_test = 10000
    d = 1
    X = torch.zeros(n_test, T, d).cuda()
    X[:, T-1, :]= torch.randn(n_test, d).cuda()
    
    for t in range(T - 2, -1, -1):
        X[:, t, :] = sampler.sample(X[:, t+1, :], t)


X = X.cpu().detach().view(n_test, T).numpy()
plot_pdf(X[:, 0])
plot_paths(X)
