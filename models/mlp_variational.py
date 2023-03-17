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
        self.fc3 = nn.Linear(hidden_dim, output_size)
        # AR more layers and more neurons per layer
        # get two values, and compute them to get mean and sdv

    def forward(self, x):
        '''Forward pass'''
        out = self.fc1(x)
        out = self.sigmoid1(out)
        out = self.fc2(out)
        out = self.sigmoid2(out)
        out = self.fc3(out)
        return out
    
