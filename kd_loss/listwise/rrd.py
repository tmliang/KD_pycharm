import torch
from kd_loss.base import BaseLoss


class RRD(BaseLoss):
    """
    SeongKu Kang, Junyoung Hwang, Wonbin Kweon and Hwanjo Yu. 2020.
    DE-RRD: A Knowledge Distillation Framework for Recommender System. In Proceedings of the CIKM.
    """
    def __init__(self, n_pos=10, n_neg=50, neg_sampler=None, s_neg=False, temperature=1):
        super().__init__(n_pos, n_neg, neg_sampler, s_neg)
        self.temperature = temperature

    def forward(self, gt, t_score, s_score):
        t_score, s_score = self.sort_scores_by_teacher(gt, t_score, s_score)
        s_score = s_score - s_score.max(dim=1, keepdim=True).values.detach()		# keep exp() stable
        loss = s_score.flip(1).logcumsumexp(1).flip(1)[:, :self.n_pos] - s_score[:, :self.n_pos]
        return torch.mean(loss.sum(1))


if __name__ == '__main__':
    gt = torch.randint(0, 10, size=(4,))
    scores = torch.rand(4, 10)
    pred = torch.rand(4, 10, requires_grad=True) * 100
    kd_loss = RRD(n_pos=2, n_neg=5)
    l = kd_loss(gt, scores, pred)
    l.backward()
    print(l)
