import torch
import torch.nn as nn
import torch.nn.functional as F


class FKD(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dist_func = args.dist_func
        self.projector = None
        if args.project:
            self.projector = nn.Linear(1536, 1536)
            nn.init.eye_(self.projector.weight)
            nn.init.zeros_(self.projector.bias)

    def forward(self, t_rep, s_rep):
        if self.projector:
            s_rep = self.projector(s_rep)
        if self.dist_func == 'mse':
            dis = F.mse_loss(t_rep, s_rep, reduction='none').sum(1) / t_rep.size(1)
        elif self.dist_func == 'norm':
            dis = torch.norm((t_rep-s_rep), dim=1)
        elif self.dist_func == 'cos':
            dis = 1 - torch.cosine_similarity(t_rep, s_rep, dim=1)
        else:
            raise NotImplementedError
        return dis.mean(), dis
