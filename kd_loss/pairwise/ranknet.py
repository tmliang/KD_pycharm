import torch
import torch.nn as nn
import torch.nn.functional as F
from kd_loss.utils import pair_minus


class RankNet(nn.Module):
    """
    Learning to rank using gradient descent. ICML, 2005.
    """
    def __init__(self, n_pos, sigma=1):
        super().__init__()
        self.sigma = sigma
        self.n_pos = n_pos
        self.loss = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, tgt_score, score):
        # get pairwise differences
        t_diff = pair_minus(tgt_score)
        s_diff = pair_minus(score)

        # only compute when i<j and i<n_pos
        mask = torch.zeros_like(t_diff[0], dtype=torch.bool)
        mask[:self.n_pos] = 1
        mask.triu_(1)

        score = torch.sigmoid(self.sigma * s_diff).masked_fill(~mask, 0)
        target = ((t_diff.sign() + 1) / 2).masked_fill(~mask, 0)

        return self.loss(score, target) / tgt_score.size(0)


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
