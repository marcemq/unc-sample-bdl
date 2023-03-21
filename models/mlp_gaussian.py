import torch
from torch import nn

class MLP_gaussian1(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super().__init__()
        # Layers
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        '''Forward pass'''
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

class MLP_gaussian2(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super().__init__()
        # Layers
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        '''Forward pass'''
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out