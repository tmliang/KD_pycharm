import torch
import torch.nn as nn
import math

eps = 1e-20


class ConcreteDropout(nn.Module):
    def __init__(self, init_p=0.1):
        super(ConcreteDropout, self).__init__()
        assert 0 < init_p < 0.5
        log_p = math.log(init_p + eps) - math.log(1 - init_p + eps)
        self.log_p = nn.Parameter(torch.tensor(log_p))      # sigmoid(log_p) = init_p

    def forward(self, x, temperature=0.1):
        u = torch.rand_like(x)
        drop_prob = (self.log_p + torch.log(u + eps) - torch.log(1 - u + eps))
        drop_prob = torch.sigmoid(drop_prob / temperature)
        scale = torch.sigmoid(self.log_p).detach()
        x = x * (1-drop_prob) / (1-scale)
        return x

    def reg(self):
        p = torch.sigmoid(self.log_p)
        return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
