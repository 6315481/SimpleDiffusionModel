import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.UNet1d import UNet1d
from datasets_generators.curves import generate_curves
from utils import plot_2d_paths
from diffusers import DDPMDiffuser1d

num_samples = 10000
batch_size = 128
num_split = 256
num_frequency = 5
radius = 10
data = generate_curves(num_samples, num_split, num_frequency, radius)
train_loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)

num_time_steps = 1000
model = UNet1d(2, 64, 2, num_time_steps).cuda()

beta = torch.linspace(10e-4, 0.02, num_time_steps).cuda()
diffusers = DDPMDiffuser1d(beta)

criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=0.0002)

epochs = 200
for epoch in range(epochs):
    for X_batch in train_loader:
        X_batch = X_batch[0].cuda()
        t = torch.randint(1, num_time_steps, (1,)).expand(X_batch.shape[0]).cuda()

        X_diffused, eps = diffusers.diffuse(X_batch, t)

        outputs = model(X_diffused, t)

        loss = criterion(outputs, eps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{epochs}], Time: {t[0].item() / num_time_steps}, Loss: {loss.item():.4f}')
