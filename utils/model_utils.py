import torch
from torch import nn

def gaussianLikelihood(y, y_hat, sigma):
    loss = (1/(2*sigma*sigma))*torch.square(y-y_hat) + 0.5* torch.log(sigma*sigma)
    return torch.mean(loss)