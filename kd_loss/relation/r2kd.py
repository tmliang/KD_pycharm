import torch
import torch.nn as nn
import torch.nn.functional as F


class R2KD(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.temperature = args.temperature
        self.sigma = args.sigma
        self.margin_ranking_loss = nn.MarginRankingLoss(margin=args.margin, reduction='mean')

    def forward(self, gt, t_logits, s_logits, dist_matrix):
        d = -dist_matrix
        d_flat = d.flatten()
        x1 = d.diagonal().repeat_interleave(len(d_flat))
        x2 = d_flat.repeat(len(d))

        ins_lv = (t_logits.argmax(1) == gt).float() + 1
        idx1 = ins_lv.repeat_interleave(len(d_flat))
        idx2 = torch.diag_embed(ins_lv).flatten().repeat(len(d))
        y = idx1 - idx2
        y[y > 0] = 1
        return self.margin_ranking_loss(x1, x2, y)