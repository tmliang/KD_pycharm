import torch
import torch.nn as nn
eps = 1e-20


class STListNet(nn.Module):
    """
    A Stochastic Treatment of Learning to Rank Scoring Functions. WSDM 2020.
    """
    def __init__(self, temperature=1):
        super().__init__()
        self.temperature = temperature

    def forward(self, score, tgt_score):
        # stochastic scores
        unif = torch.rand(score.size(), device=score.device)
        gumbel = -torch.log(-torch.log(unif + eps) + eps)  # Sample from gumbel distribution
        score = (score + gumbel) / self.temperature

        return -torch.sum(tgt_score.softmax(dim=1) * score.log_softmax(dim=1), dim=1).mean()