import torch
import torch.nn as nn
import torch.nn.functional as F


class R2KD(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.temperature = args.temperature
        self.sigma = args.sigma
        self.margin_ranking_loss = nn.MarginRankingLoss(margin=args.margin, reduction='mean')

    def forward(self, gt, t_logits, s_logits, dis):
        d = -dis
        n = d.size(0)
        x1 = d.view(-1, 1).repeat(1, n)
        x2 = d.view(1, -1).repeat(n, 1)
        gt_rank = (t_logits.argsort(1) == gt.view(-1, 1)).float().argmax(1)
        idx1 = gt_rank.view(-1, 1).repeat(1, n)
        idx2 = gt_rank.view(1, -1).repeat(n, 1)
        y = idx1 - idx2
        tgt = y > 0
        # y[tgt] = 1
        if tgt.sum() > 0:
            return self.margin_ranking_loss(x1[tgt], x2[tgt], y[tgt])
        else:
            return 0
