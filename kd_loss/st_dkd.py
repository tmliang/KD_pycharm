import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import Sampler
eps = 1e-20


class STDecoupled_KD(nn.Module):
    def __init__(self, n_pos=10, n_neg=50, neg_sampler=None, temperature=1, alpha=1, beta=8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.n_pos = n_pos
        self.temperature = temperature
        self.Sampler = Sampler(n_neg, actor=neg_sampler)

    def forward(self, gt, t_score, s_score):
        # stochastic scores
        unif = torch.rand(s_score.size(), device=s_score.device)
        gumbel = -torch.log(-torch.log(unif + eps) + eps)  # Sample from gumbel distribution
        s_score = (s_score + gumbel) / self.temperature

        # target
        gt_mask = torch.zeros_like(t_score).scatter_(1, gt.unsqueeze(1), 1).bool()
        t_pred = F.softmax(t_score, dim=1)
        s_pred = F.softmax(s_score, dim=1)
        t_tgt = self.cat_target(t_pred, gt_mask)
        s_tgt = self.cat_target(s_pred, gt_mask)
        tckd = -torch.sum(t_tgt * s_tgt.log(), dim=1).mean()

        # positive
        sorted_ind = torch.sort(t_score.masked_fill(gt_mask, float('-inf')), descending=True).indices
        pos_list = sorted_ind[:, :self.n_pos]
        t_pos = t_score.gather(1, pos_list)
        s_pos = s_score.gather(1, pos_list)
        pckd = -torch.sum(t_pos.softmax(dim=1) * s_pos.log_softmax(dim=1), dim=1).mean()

        # negative
        n_gt = 1 if len(gt.size()) == 1 else gt.size(1)
        neg_list = self.Sampler(sorted_ind[:, self.n_pos:-n_gt])
        t_neg = t_score.gather(1, neg_list)
        s_neg = s_score.gather(1, neg_list)
        nckd = -torch.sum(t_neg.softmax(dim=1) * s_neg.log_softmax(dim=1), dim=1).mean()
        return tckd + self.alpha * pckd + self.beta * nckd

    def cat_target(self, score, mask):
        pt = (score * mask).sum(1, keepdims=True)
        pnt = (score * ~mask).sum(1, keepdims=True)
        return torch.cat([pt, pnt], dim=1)