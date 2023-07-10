import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaKD(nn.Module):
    def __init__(self, temperature=1):
        super().__init__()
        self.temperature = temperature

    def forward(self, gt, t_score, s_score):
        return F.kl_div(
            torch.log_softmax(s_score / self.temperature, dim=1),
            torch.softmax(t_score / self.temperature, dim=1),
            reduction="batchmean"
        ) * self.temperature ** 2