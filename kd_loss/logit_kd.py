import torch
import torch.nn as nn
import torch.nn.functional as F


class Logit_KD(nn.Module):
    """
    Zhe Cao, Tao Qin, Tie-Yan Liu, Ming-Feng Tsai, and Hang Li. 2007.
    Learning to Rank: From Pairwise Approach to Listwise Approach. In Proceedings of the 24th ICML. 129–136.
    """
    def __init__(self, temperature=1):
        super().__init__()
        self.temperature = temperature

    def forward(self, gt, t_score, s_score):
        return F.kl_div(
            torch.log_softmax(s_score / self.temperature, dim=1),
            torch.softmax(t_score / self.temperature, dim=1),
            reduction="batchmean"
        ) * self.temperature ** 2