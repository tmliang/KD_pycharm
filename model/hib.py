import torch
import torch.nn as nn
import torch.nn.functional as F


class StochasticLayer(nn.Module):
    def __init__(self, hib_factor, hidden_dim, dropout=0.1):
        super().__init__()
        assert not hidden_dim % hib_factor
        self.mu_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // hib_factor),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // hib_factor, hidden_dim),
        )
        self.logsigma_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // hib_factor),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // hib_factor, hidden_dim),
        )
        self.apply(self.init_weights)

    def init_weights(self, m, std=1e-3):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=std)
            torch.nn.init.normal_(m.bias, std=std)
            m.weight.data = torch.clamp(m.weight.data, min=-2 * std, max=2 * std)
            m.bias.data = torch.clamp(m.bias.data, min=-2 * std, max=2 * std)

    def forward(self, x):
        mu = self.mu_layer(x)
        # mu = F.normalize(mu, p=2, dim=1)
        sigma = torch.exp(self.logsigma_layer(x))
        return mu, sigma


class HedgedInstanceEmbedding(nn.Module):
    def __init__(self, hib_factor, hidden_dim, dropout=0.1, a=1, b=0):
        super().__init__()
        self.stochastic_layer = StochasticLayer(hib_factor, hidden_dim, dropout)
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float))

    def forward(self, x, num_sample):
        mu, sigma = self.stochastic_layer(x)
        eps = torch.randn(mu.size(0), num_sample, mu.size(1)).to(mu)
        z = eps * sigma.unsqueeze(1) + mu.unsqueeze(1)
        return z, (mu, sigma)

    def euclid_loss(self, x1, x2):
        return -self.a * (x1.unsqueeze(2) - x2.unsqueeze(0).unsqueeze(0)).norm(dim=-1).mean(1) + self.b

    def product_loss(self, x1, x2):
        return self.a * (x1@x2.t()).mean(1) + self.b