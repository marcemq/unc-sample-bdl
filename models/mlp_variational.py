import torch
from torch import nn
from bayesian_torch.layers import LinearReparameterization

class MLP_variational(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(MLP_variational, self).__init__()
        # Linear rep layer: with prior standard normal dist
        self.linearRep1 = LinearReparameterization(
            in_features = input_size,
            out_features = hidden_dim,
            prior_mean = 0,
            prior_variance = 1,
            posterior_mu_init = 0,
            posterior_rho_init = -4,
        )
        self.relu1 = nn.ReLU()
        # Linear rep layer: with prior standard normal dist
        self.linearRep2 = LinearReparameterization(
            in_features = hidden_dim,
            out_features = output_size,
            prior_mean = 0,
            prior_variance = 1,
            posterior_mu_init = 0,
            posterior_rho_init = -4,
        )
        self.loss_fun = nn.MSELoss()

    def forward(self, x, y_gt, mc_its=1):
        '''Forward pass'''
        pred_ = []
        kl_ = []

        for mc_run in range(mc_its):
            kl_sum = 0

            x, kl = self.linearRep1(x)
            kl_sum += kl
            x = self.relu1(x)
            pred, kl = self.linearRep2(x)
            kl_sum += kl
            pred_.append(pred)
            kl_.append(kl_sum)

        y_pred    = torch.mean(torch.stack(pred_), dim=0)
        kl_loss = torch.mean(torch.stack(kl_), dim=0)
        # compute nll loss
        nll_loss = self.loss_fun(pred, y_gt)
        return y_pred, nll_loss, kl_loss
    
