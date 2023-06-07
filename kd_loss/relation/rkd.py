import torch
import torch.nn as nn
import torch.nn.functional as F


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


class RKD(nn.Module):
    """
    Relational Knowledge Distillation. CVPR 2019.
    """
    def __init__(self, lambda_dist=1, lambda_angle=2):
        super().__init__()
        self.lambda_dist = lambda_dist
        self.lambda_angle = lambda_angle

    def forward(self, t_rep, s_rep):
        dist_loss = self.distance_wise_loss(t_rep, s_rep)
        angle_loss = self.angle_wise_loss(t_rep, s_rep)
        return self.lambda_dist * dist_loss + self.lambda_angle * angle_loss

    def distance_wise_loss(self, t_rep, s_rep):
        with torch.no_grad():
            td = pdist(t_rep, squared=False)
            mean_td = td[td > 0].mean()
            td = td / mean_td
        sd = pdist(s_rep, squared=False)
        mean_sd = sd[sd > 0].mean()
        sd = sd / mean_sd
        return F.smooth_l1_loss(sd, td, reduction='mean')

    def angle_wise_loss(self, t_rep, s_rep):
        with torch.no_grad():
            td = (t_rep.unsqueeze(0) - t_rep.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (s_rep.unsqueeze(0) - s_rep.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
        return F.smooth_l1_loss(s_angle, t_angle, reduction='mean')