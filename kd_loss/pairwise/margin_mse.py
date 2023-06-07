import torch
import torch.nn as nn
from kd_loss.utils import pair_minus


class MarginMSE(nn.Module):
    """
    Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation. 2020.
    https://arxiv.org/abs/2010.02666.
    """
    def __init__(self, margin=0):
        super().__init__()
        self.margin = margin
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, score, tgt_score):
        # get pairwise differences
        t_diff = pair_minus(tgt_score)
        s_diff = pair_minus(score)
        # ignore the neg-neg pairs
        mask = torch.zeros_like(t_diff[0], dtype=torch.bool)
        mask[:self.n_pos] = 1
        mask.triu_(1)
        t_diff.masked_fill_(~mask, 0)
        s_diff.masked_fill_(~mask, 0)
        return self.mse_loss(t_diff, s_diff)


# if __name__ == '__main__':
#     import torch.optim as optim
#     N = 10
#     gt = torch.randint(0, N, size=(2,))
#     scores = torch.rand(2, N) * 100
#     pred = torch.rand(2, N) * 100
#     pred.requires_grad = True
#     ori = pred.clone()
#     kd_loss = MarginMSE(n_pos=3, n_neg=3, neg_sampler='random')
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
