import torch
import torch.nn as nn
import torch.nn.functional as F


class StochasticLayer(nn.Module):
    def __init__(self, hib_factor, dim, dropout=0.1):
        super().__init__()
        assert not dim % hib_factor
        hidden_dim = dim // hib_factor
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.apply(self.init_weights)

    def init_weights(self, m, std=1e-3):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=std)
            torch.nn.init.normal_(m.bias, std=std)
            m.weight.data = torch.clamp(m.weight.data, min=-2 * std, max=2 * std)
            m.bias.data = torch.clamp(m.bias.data, min=-2 * std, max=2 * std)


class MuLayer(StochasticLayer):
    def __init__(self, hib_factor, dim, dropout=0.1):
        super().__init__(hib_factor, dim, dropout)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.fc2(x)
        return x


class LogSigmaLayer(StochasticLayer):
    def __init__(self, hib_factor, dim, dropout=0.1):
        super().__init__(hib_factor, dim, dropout)

    def forward(self, x):
        x = self.drop(self.fc1(x))
        x = self.fc2(x)
        return x


class HedgedInstanceEmbedding(nn.Module):
    def __init__(self, hib_factor, dim, dropout=0.1):
        super().__init__()
        self.mu_layer = MuLayer(hib_factor, dim, dropout)
        self.logsigma_layer = LogSigmaLayer(hib_factor, dim, dropout)

    def forward(self, x):
        mu = self.mu_layer(x)
        sigma = torch.exp(self.logsigma_layer(x))
        return [mu, sigma]

    def sample(self, proto, num_sample):
        mu, sigma = proto
        eps = torch.randn(mu.size(0), num_sample, mu.size(1)).to(mu)
        z = eps * sigma.unsqueeze(1) + mu.unsqueeze(1)
        return z
