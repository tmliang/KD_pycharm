import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoupled_KD(nn.Module):
    """
    Decoupled Knowledge Distillation. CVPR 2022.
    """
    def __init__(self, temperature=1, alpha=1, beta=8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def forward(self, gt, t_score, s_score):
        t_score /= self.temperature
        s_score /= self.temperature
        gt_mask = torch.zeros_like(t_score).scatter_(1, gt.unsqueeze(1), 1).bool()
        t_pred = F.softmax(t_score, dim=1)
        s_pred = F.softmax(s_score, dim=1)
        t_tgt = self.cat_target(t_pred, gt_mask)
        s_tgt = self.cat_target(s_pred, gt_mask)
        tckd = F.kl_div(s_tgt.log(), t_tgt, reduction="batchmean")
        nckd = F.kl_div(
            torch.log_softmax(s_score.masked_fill(gt_mask, -1000), dim=1),
            torch.softmax(t_score.masked_fill(gt_mask, -1000), dim=1),
            reduction="batchmean"
        )
        return (self.alpha * tckd + self.beta * nckd) * self.temperature ** 2

    def cat_target(self, score, mask):
        pt = (score * mask).sum(1, keepdims=True)
        pnt = (score * ~mask).sum(1, keepdims=True)
        return torch.cat([pt, pnt], dim=1)