import torch
from kd_loss.base import BaseLoss
eps = 1e-20


class STListNet(BaseLoss):
    """
    Bruch, Sebastian and Han, Shuguang and Bendersky, Michael and Najork, Marc. 2020.
    A Stochastic Treatment of Learning to Rank Scoring Functions. WSDM.
    """
    def __init__(self, n_pos=10, n_neg=50, neg_sampler=None, s_neg=False, temperature=1):
        super().__init__(n_pos, n_neg, neg_sampler, s_neg)
        self.s_neg = s_neg
        self.neg_sampler = neg_sampler
        self.temperature = temperature

    def forward(self, gt, t_score, s_score):
        t_score, s_score = self.sort_scores_by_teacher(gt, t_score, s_score)

        # stochastic scores
        unif = torch.rand(s_score.size(), device=s_score.device)
        gumbel = -torch.log(-torch.log(unif + eps) + eps)  # Sample from gumbel distribution
        s_score = (s_score + gumbel) / self.temperature

        return -torch.sum(t_score.softmax(dim=1) * s_score.log_softmax(dim=1), dim=1).mean()