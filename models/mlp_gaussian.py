import torch
from torch import nn

class MLP_gaussian(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super().__init__()
        # Layers
        self.fc = nn.Linear(input_size, hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''Forward pass'''
        out = self.fc(x)
        out = self.sigmoid(out)
        return out
    
