import torch
from torch import nn

class NNModel(nn.Module):
    def __init__(self, input_features, hidden_unit=8):
        super().__init__()
        self.layer1= nn.Linear(in_features=input_features, out_features=64)
        self.layer2= nn.Linear(in_features=64, out_features= 64)
        self.layer3= nn.Linear(in_features=64, out_features=1)
        self.relu= nn.ReLU()
    def forward(self, x):
        return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))