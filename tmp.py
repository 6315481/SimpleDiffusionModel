import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from noise_models import NoiseModel1Rank
from models.models import ReLUFNN
from diffusers import DDPMDiffuser


n = 5000
p = [0.25, 0.25, 0.25, 0.25]
X_train = np.random.choice(a = [-10, -2, 2, 10], size = n, p = p)
X_train = torch.from_numpy(X_train).view(-1, 1).cuda()

T = 1000
beta = torch.linspace(10e-4, 0.02, T).cuda()
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0).cuda()

diffuser = DDPMDiffuser(beta, device='cuda:0')

# モデル、損失関数、最適化手法の定義
model = ReLUFNN([2, 100, 100, 100, 1]).cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ノイズ関数の定義
noise_model = NoiseModel1Rank(model, T, device='cuda:0')

# 学習
epochs = 20000
for epoch in range(epochs):
    t = torch.randint(1, T, (1,)).cuda()

    X_diffused, eps = diffuser.diffuse(X_train, t)
    
    # 順伝播
    outputs = noise_model(X_diffused, t)
    
    # 損失計算
    loss = criterion(outputs, eps)
    
    # 逆伝播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 500 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Time: {t / T}, Loss: {loss.item():.4f}')


torch.save(model, "model_weight.pth")

