import torch
import torch.nn as nn
from kd_loss.utils import pair_minus


class LambdaLoss(nn.Module):
    """
    Wang, Xuanhui and Li, Cheng and Golbandi, Nadav and Bendersky, Michael and Najork, Marc. 2018.
    The LambdaLoss Framework for Ranking Metric Optimization. ICIKM. 1313–1322.
    """
    def __init__(self, weight_scheme='ndcg', sigma=1, tau=1, temperature=1):
        super().__init__()
        self.sigma = sigma
        self.tau = tau
        self.temperature = temperature
        if weight_scheme == 'ndcg':
            self.weight_scheme = self.ndcgloss
        elif weight_scheme == 'ndcg1':
            self.weight_scheme = self.ndcgloss1
        elif weight_scheme == 'ndcg2':
            self.weight_scheme = self.ndcgloss2
        elif weight_scheme == 'ndcg2++':
            self.weight_scheme = self.ndcgloss2pp
        else:
            raise NotImplementedError

    def forward(self, tgt_score, score):
        tgt_softmax = torch.softmax(tgt_score / self.temperature, dim=-1)
        D = (1 / torch.log2(torch.arange(2, tgt_score.size(1) + 2, device=tgt_score.device))).unsqueeze(0)  # inverse discounts
        idcg = torch.sum((2 ** tgt_softmax - 1) * D, dim=1, keepdim=True)   # ideal dcg
        G = (2 ** tgt_softmax - 1) / idcg  # normalised gains

        weight = self.weight_scheme(G, D)
        loss = weight * torch.log(torch.sigmoid(self.sigma * pair_minus(score)).clamp(min=1e-10))

        # mask
        mask = torch.ones_like(loss[0], dtype=torch.bool)
        if self.weight_scheme != 'ndcg1':
            mask.triu_(1)
        return -torch.sum(loss.masked_select(mask)) / loss.size(0)

    def ndcgloss(self, G, D):
        return torch.abs(pair_minus(G) * pair_minus(D))

    def ndcgloss1(self, G, D):
        return (G*D).unsqueeze(2)

    def ndcgloss2(self, G, D):
        relative_pos = torch.abs(pair_minus(torch.arange(1, G.size(1) + 1, device=G.device)))
        delta = torch.abs(D[0, relative_pos-1] - D[0, relative_pos])
        delta.diagonal().zero_()
        return delta * torch.abs(pair_minus(G))

    def ndcgloss2pp(self, G, D):
        return (self.tau * self.ndcgloss2(G, D) + torch.abs(pair_minus(D))) * torch.abs(pair_minus(G))


# if __name__ == '__main__':
#     import torch.optim as optim
#     N = 6
#     gt = torch.randint(0, N, size=(2,))
#     scores = torch.rand(2, N)
#     pred = torch.rand(2, N)
#     pred.requires_grad = True
#     ori = pred.clone()
#     kd_loss = LambdaLoss(n_pos=3, n_neg=3, weight_scheme='ndcg2++', sigma=1., temperature=1.)
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