import torch
import torch.nn as nn
import torch.nn.functional as F
from cs336_basics.nn_utils import cross_entropy

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        print("ToyModel.fc1 = ",x)
        x = self.ln(x)
        print("ToyModel.ln = ",x.dtype)
        x = self.fc2(x)
        return x

x = torch.normal(0,1,size=(10,)).to('mps')
y = torch.normal(0,1,size=(10,)).to('mps')

model = ToyModel(10, 10).to('mps')

for name, param in model.named_parameters():
    print(f"{name}: {param.dtype}")

model.train()
with torch.autocast(device_type="mps", dtype=torch.bfloat16):
    logits = model(x)
    print("logits = ", logits.dtype)
    loss = F.cross_entropy(logits, y)
    print("loss type = ",loss.dtype)
    loss.backward()
    for name, param in model.named_parameters():
        print(f"{name}: {param.grad.dtype}")


