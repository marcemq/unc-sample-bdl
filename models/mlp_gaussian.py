import torch
from torch import nn

class MLP_gaussian1(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(MLP_gaussian1, self).__init__()
        # Layers
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        '''Forward pass'''
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        return out

class MLP_gaussian2(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(MLP_gaussian2, self).__init__()
        # Layers
        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        '''Forward pass'''
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        return out
