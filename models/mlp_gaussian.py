import torch
from torch import nn

class MLP_gaussian(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super().__init__()
        # Layers
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.sigmoid1 = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid2 = nn.Sigmoid()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid3 = nn.Sigmoid()
        self.fc4 = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        '''Forward pass'''
        out = self.fc1(x)
        out = self.sigmoid1(out)
        out = self.fc2(out)
        out = self.sigmoid2(out)
        out = self.fc3(out)
        out = self.sigmoid3(out)
        out = self.fc4(out)
        return out