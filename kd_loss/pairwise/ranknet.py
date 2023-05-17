import torch
import torch.nn.functional as F
from kd_loss.utils import pair_minus
from kd_loss.base import BaseLoss


class RankNet(BaseLoss):
    """
    Learning to rank using gradient descent. ICML, 2005.
    """
    def __init__(self, n_pos=10, n_neg=50, neg_sampler=None, sigma=1.):
        super().__init__(n_pos, n_neg, neg_sampler)
        self.neg_sampler = neg_sampler
        self.sigma = sigma

    def forward(self, gt, t_score, s_score):
        t_score, s_score = self.sort_scores_by_teacher(gt, t_score, s_score)

        # get pairwise differences
        t_diff = pair_minus(t_score)
        s_diff = pair_minus(s_score)

        # only compute when i<j and i<n_pos
        mask = torch.zeros_like(t_diff[0], dtype=torch.bool)
        mask[:self.n_pos] = 1
        mask.triu_(1)

        score = torch.sigmoid(self.sigma * s_diff).masked_fill(~mask, 0)
        target = ((t_diff.sign() + 1) / 2).masked_fill(~mask, 0)

        return F.binary_cross_entropy(input=score, target=target, reduction='sum') / t_score.size(0)


# if __name__ == '__main__':
#     import torch.optim as optim
#     N = 10
#     gt = torch.randint(0, N, size=(2,))
#     scores = torch.rand(2, N) * 100
#     pred = torch.rand(2, N) * 100
#     pred.requires_grad = True
#     ori = pred.clone()
#     kd_loss = RankNet(n_pos=3, n_neg=3, neg_sampler='random', sigma=0.5)
#     optimizer = optim.Adam([pred], lr=0.01)
#     for epoch in range(10000):
#         optimizer.zero_grad()
#         loss = kd_loss(gt, scores, pred)
#         loss.backward()
#         optimizer.step()
#         if epoch % 1000 == 0:
#             print(f"{epoch} | loss: {loss.item()}")
#     kd_loss(gt, scores, pred*100)
#     print("original:", ori.sort(1, descending=True)[1].tolist())
#     print("ground-truth:", scores.sort(1, descending=True)[1].tolist())
#     print("updated:", pred.sort(1, descending=True)[1].tolist())
