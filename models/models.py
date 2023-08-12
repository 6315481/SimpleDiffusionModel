import torch

class ReLUFNN(torch.nn.Module):
    def __init__(self, layer_dims):
        super(ReLUFNN, self).__init__()

        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(torch.nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i != len(layer_dims) - 2:
                layers.append(torch.nn.ReLU())

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)