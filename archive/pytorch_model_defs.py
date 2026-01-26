import torch
import torch.nn.functional as F

class my_first_pytorch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Layers
        self.fc1 = torch.nn.Linear(in_features=28*28, out_features=512)
        self.fc2 = torch.nn.Linear(in_features=512, out_features=256)
        self.fc3 = torch.nn.Linear(in_features=256, out_features=10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
