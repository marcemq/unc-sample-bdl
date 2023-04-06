import torch
from torch import nn

def gaussianLikelihood(y, y_hat, sigma):
    loss = (1/(2*sigma*sigma))*torch.square(y-y_hat) + 0.5* torch.log(sigma*sigma)
    return torch.mean(loss)

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
        pred_mean, rho = out[:, :1], out[:, 1:]
        # Sigma reparametrization from Weight Uncertainty in NN
        sigma = torch.log1p(torch.exp(rho))
        # compute loss
        loss = gaussianLikelihood(y_gt, pred_mean, sigma)
        return pred_mean, sigma, loss
