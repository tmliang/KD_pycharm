import torch
import torch.nn as nn
import math


class BayesianMask(nn.Module):
    def __init__(self, hidden_size, mu=0, sigma=1, temperature=0.1):
        super(BayesianMask, self).__init__()
        self.prior_mu = mu
        self.prior_log_sigma = math.log(sigma)
        self.mu = nn.Parameter(torch.Tensor(1, hidden_size))
        self.log_sigma = nn.Parameter(torch.Tensor(1, hidden_size))
        self.temperature = temperature
        self.init_weights()

    def init_weights(self):
        std = 1. / math.sqrt(self.mu.size(1))
        self.mu.data.uniform_(-std, std)
        self.log_sigma.data.fill_(self.prior_log_sigma)

    def forward(self, x):
        u = torch.randn_like(x)
        m = self.mu + torch.exp(self.log_sigma) * u
        m = torch.sigmoid(m / self.temperature)
        scale = torch.sigmoid(self.mu.mean().detach())
        x = x * m / scale
        return x

    def reg(self):
        return torch.sum(self.log_sigma - self.prior_log_sigma + (
                    math.exp(self.prior_log_sigma) ** 2 + (self.prior_mu - self.mu) ** 2) / (
                                     2 * torch.exp(self.log_sigma) ** 2) - 0.5)
