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
            return torch.sum((t_rep.unsqueeze(1) - s_rep.unsqueeze(0))**2, dim=-1)
        elif self.dist_func == 'norm':
            return torch.norm(t_rep.unsqueeze(1) - s_rep.unsqueeze(0), dim=-1)
        elif self.dist_func == 'cos':
            return torch.cosine_similarity(t_rep.unsqueeze(1), s_rep.unsqueeze(0), dim=2)
        else:
            raise NotImplementedError

        # diag = d.diag()
        # non_diag = d.masked_select(~torch.eye(d.size(0), dtype=torch.bool))
        # return diag, non_diag
