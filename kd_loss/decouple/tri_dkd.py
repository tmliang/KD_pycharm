import torch
import torch.nn as nn
import torch.nn.functional as F
from kd_loss.base import Sampler
eps = 1e-20


class TriDecoupled_KD(nn.Module):
    def __init__(self, n_pos=10, n_neg=50, neg_sampler=None, temperature=1, alpha=5, beta=1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.n_pos = n_pos
        self.temperature = temperature
        self.Sampler = Sampler(n_neg, actor=neg_sampler)

    def forward(self, gt, t_score, s_score):
        t_score /= self.temperature
        s_score /= self.temperature

        # target
        gt_mask = torch.zeros_like(t_score).scatter_(1, gt.unsqueeze(1), 1).bool()
        t_pred = F.softmax(t_score, dim=1)
        s_pred = F.softmax(s_score, dim=1)
        t_tgt = self.cat_target(t_pred, gt_mask)
        s_tgt = self.cat_target(s_pred, gt_mask)
        tckd = F.kl_div(s_tgt.log(), t_tgt, reduction="batchmean")

        # positive
        sorted_ind = torch.sort(t_score.masked_fill(gt_mask, float('-inf')), descending=True).indices
        pos_list = sorted_ind[:, :self.n_pos]
        t_pos = t_score.gather(1, pos_list)
        s_pos = s_score.gather(1, pos_list)
        pckd = F.kl_div(
            torch.log_softmax(s_pos, dim=1),
            torch.softmax(t_pos, dim=1),
            reduction="batchmean"
        )

        # negative
        n_gt = 1 if len(gt.size()) == 1 else gt.size(1)
        neg_list = self.Sampler(sorted_ind[:, self.n_pos:-n_gt])
        t_neg = t_score.gather(1, neg_list)
        s_neg = s_score.gather(1, neg_list)
        nckd = F.kl_div(
            torch.log_softmax(s_neg, dim=1),
            torch.softmax(t_neg, dim=1),
            reduction="batchmean"
        )
        return (tckd + self.alpha * pckd + self.beta * nckd) * self.temperature ** 2

    def cat_target(self, score, mask):
        pt = (score * mask).sum(1, keepdims=True)
        pnt = (score * ~mask).sum(1, keepdims=True)
        return torch.cat([pt, pnt], dim=1)