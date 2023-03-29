import torch
from torch import nn

def gaussianLikelihood(y, pred, sigma):
    loss = (1/(2*sigma*sigma))*torch.norm(y-pred, p=2) + 0.5* torch.log(sigma*sigma)
    return loss

class MLP1(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(MLP1, self).__init__()
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

class MLP2(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(MLP2, self).__init__()
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

class MLP_gaussian(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(MLP_gaussian, self).__init__()
        # Layers
        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        # added output for sigma
        self.fc2 = nn.Linear(hidden_dim, output_size + 1)

    def forward(self, x, y_gt):
        '''Forward pass'''
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        pred, sigma = out[:, :1], out[:, 1:]
        # compute loss
        loss = gaussianLikelihood(y_gt, pred, sigma)
        print(loss.shape)
        return pred, sigma, loss
            