import torch
import torch.nn as nn
import torch.nn.functional as F
from kd_loss.utils import pair_minus


class LambdaRank(nn.Module):
    """
    Learning to Rank with Nonsmooth Cost Functions. NIPS. 2006.
    """
    def __init__(self, n_pos, sigma=1, temperature=1):
        super().__init__()
        self.sigma = sigma
        self.n_pos = n_pos
        self.temperature = temperature

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
        delta = self.delta_ndcg(tgt_score * self.temperature).masked_fill(~mask, 0)

        return F.binary_cross_entropy(input=score, target=target, weight=delta, reduction='sum') / score.size(0)

    def delta_ndcg(self, score):
        D = (1 / torch.log2(torch.arange(2, score.size(1) + 2, device=score.device))).unsqueeze(0)  # discounts
        idcg = torch.sum((2 ** score - 1) * D, dim=1, keepdim=True)   # ideal dcg
        G = (2 ** score - 1) / idcg  # normalised gains
        return torch.abs(pair_minus(G) * pair_minus(D))


# if __name__ == '__main__':
#     import torch.optim as optim
#     N = 10
#     gt = torch.randint(0, N, size=(2,))
#     scores = torch.rand(2, N) * 100
#     pred = torch.rand(2, N) * 100
#     pred.requires_grad = True
#     ori = pred.clone()
#     kd_loss = LambdaRank(n_pos=3, n_neg=3, neg_sampler='zipfs', sigma=1.)
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
#     print("ground-truth:", gt.tolist())
#     print("teacher:", scores.sort(1, descending=True)[1].tolist())
#     print("updated:", pred.sort(1, descending=True)[1].tolist())