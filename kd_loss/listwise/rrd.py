import torch
import torch.nn as nn


class RRD(nn.Module):
    """
    SeongKu Kang, Junyoung Hwang, Wonbin Kweon and Hwanjo Yu. 2020.
    DE-RRD: A Knowledge Distillation Framework for Recommender System. In Proceedings of the CIKM.
    """
    def __init__(self, temperature=1):
        super().__init__()
        self.temperature = temperature

    def forward(self, score, tgt_score=None):
        score = score - score.max(dim=1, keepdim=True).values.detach()		# keep exp() stable
        loss = score.flip(1).logcumsumexp(1).flip(1)[:, :self.n_pos] - score[:, :self.n_pos]
        return torch.mean(loss.sum(1))


# if __name__ == '__main__':
#     gt = torch.randint(0, 10, size=(4,))
#     scores = torch.rand(4, 10)
#     pred = torch.rand(4, 10, requires_grad=True) * 100
#     kd_loss = RRD(n_pos=2, n_neg=5)
#     l = kd_loss(gt, scores, pred)
#     l.backward()
#     print(l)
