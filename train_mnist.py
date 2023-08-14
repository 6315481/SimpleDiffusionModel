import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from noise_models import NoiseModel2Rank
from diffusers import DDPMDiffuser2d
from models.UNet2d import UNet2d

#データ生成
batch_size = 128
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32))
])
mnist_data = torchvision.datasets.MNIST(root="./datasets", train=True, transform=transform)
data_loader = torch.utils.data.DataLoader(mnist_data, batch_size, shuffle=True, num_workers=0)

#タイムステップ
T = 1000

#モデル定義
model = UNet2d(1, 64, 1, T).cuda()

#ディフューザー定義
beta = torch.linspace(10e-4, 0.02, T).cuda()

diffuser = DDPMDiffuser2d(beta)

criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=0.0002)

# 学習
epochs = 200
for epoch in range(epochs):
    for i, X_batch in enumerate(data_loader):
        X_batch = X_batch[0].cuda()
        t = torch.randint(1, T, (1,)).expand(X_batch.shape[0]).cuda()

        X_diffused, eps = diffuser.diffuse(X_batch, t)
    
        # 順伝播
        outputs = model(X_diffused, t)
    
        # 損失計算
        loss = criterion(outputs, eps)
    
        # 逆伝播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 50 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Time: {t[0].item() / T}, Loss: {loss.item():.4f}')
    
    if epoch % 5 == 0:
        torch.save(model, f'./ckpts/mnist/model_weight_{epoch}.pth')
    