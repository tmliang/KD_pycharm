import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BayesianLayer(nn.Module):
    def __init__(self, d_in, d_out, bias=True):
        super().__init__()
        self.mu = nn.Parameter(torch.Tensor(d_out, d_in))
        self.log_sigma = nn.Parameter(torch.Tensor(d_out, d_in))
        self.bias = bias
        if bias:
            self.bias_mu = nn.Parameter(torch.Tensor(d_out))
            self.bias_log_sigma = nn.Parameter(torch.Tensor(d_out))

    def forward(self, x):
        weight = self.mu + torch.exp(self.log_sigma) * torch.randn_like(self.mu) * 0.1
        bias = None
        if self.bias:
            bias = self.bias_mu + torch.exp(self.bias_log_sigma) * torch.randn_like(self.bias_mu)
        return F.linear(x, weight, bias)


class BNN(nn.Module):
    def __init__(self, hidden_size, ds_factor=8, mu=0, sigma=1, temperature=0.1, dropout=0.1):
        super().__init__()
        self.prior_mu = mu
        self.prior_log_sigma = math.log(sigma)
        self.bnn_layer = nn.Sequential(
            BayesianLayer(hidden_size, hidden_size//ds_factor),
            nn.ReLU(),
            nn.Dropout(dropout),
            BayesianLayer(hidden_size//ds_factor, hidden_size),
        )
        self.ln = nn.LayerNorm(hidden_size)
        self.gate = nn.Parameter(torch.Tensor(1))
        self.init_weights()

    def init_weights(self, std=1e-3):
        for name, param in self.bnn_layer.named_parameters():
            if 'mu' in name:
                param.data.uniform_(-std, std)
            if 'sigma' in name:
                param.data.fill_(self.prior_log_sigma)
        self.ln.bias.data.zero_()
        self.ln.weight.data.fill_(1.0)
        self.gate.data.uniform_(-std, std)

    def forward(self, x):
        xu = self.bnn_layer(x)
        x = x + torch.tanh(self.gate) * xu
        return x

    def reg(self):
        loss = 0
        for m in self.modules():
            if isinstance(m, BayesianLayer):
                loss += self._kl(m.mu, m.log_sigma)
                if m.bias:
                    loss += self._kl(m.bias_mu, m.bias_log_sigma)
        return loss

    def _kl(self, mu, log_sigma):
        return torch.mean(
            log_sigma - self.prior_log_sigma + (math.exp(self.prior_log_sigma) ** 2 + (self.prior_mu - mu) ** 2) / (
                        2 * torch.exp(log_sigma) ** 2) - 0.5)
