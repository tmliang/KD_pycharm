import torch
from kd_loss.utils import pair_minus
from kd_loss.base import BaseLoss


class LambdaLoss(BaseLoss):
    """
    Wang, Xuanhui and Li, Cheng and Golbandi, Nadav and Bendersky, Michael and Najork, Marc. 2018.
    The LambdaLoss Framework for Ranking Metric Optimization. ICIKM. 1313–1322.
    """
    def __init__(self, n_pos=10, n_neg=50, neg_sampler=None, weight_scheme='ndcg1', sigma=1., mu=1.):
        super().__init__(n_pos, n_neg, neg_sampler)
        self.weight_scheme = weight_scheme
        self.sigma = sigma
        self.mu = mu

    def forward(self, gt, t_score, s_score):
        y, s = self.sort_scores_by_teacher(gt, t_score, s_score)
        # y = self.quantize(y.gather(1, ind), rounding=True)
        # s = self.quantize(s.gather(1, ind), rounding=False)

        D = (1 / torch.log2(torch.arange(2, y.size(1) + 2, device=y.device))).unsqueeze(0)  # inverse discounts
        idcg = torch.sum((2 ** y - 1) * D, dim=1, keepdim=True)   # ideal dcg
        G = (2 ** y - 1) / idcg  # normalised gains

        # Weight scheme
        if self.weight_scheme == 'ndcg1':
            weight = self.ndcgloss1(G, D)
        elif self.weight_scheme == 'ndcg2':
            weight = self.ndcgloss2(G, D)
        elif self.weight_scheme == 'ndcg2++':
            weight = self.ndcgloss2pp(G, D)
        else:
            raise NotImplementedError

        # lambda
        lam = (torch.sigmoid(self.sigma * pair_minus(s)) ** weight).clamp(min=1e-10)
        loss = torch.log(lam)
        # mask
        mask = torch.ones_like(lam[0], dtype=torch.bool)
        mask[self.n_pos:, self.n_pos:] = 0
        if self.weight_scheme != 'ndcg1':
            mask.triu_(1)
        return -torch.mean(torch.sum(loss.masked_fill(~mask, 0), dim=(1, 2)))

    def ndcgloss1(self, G, D):
        return (G*D).unsqueeze(2)

    def ndcgloss2(self, G, D):
        relative_pos = torch.abs(pair_minus(torch.arange(1, G.size(1) + 1, device=G.device)))
        delta = torch.abs(D[0, relative_pos-1] - D[0, relative_pos])
        delta.diagonal().zero_()
        return delta * torch.abs(pair_minus(G))

    def ndcgloss2pp(self, G, D):
        return (self.mu * self.ndcgloss2(G, D) + torch.abs(pair_minus(D))) * torch.abs(pair_minus(G))

    def quantize(self, score, low=0, high=16, rounding=False):
        # Quantization -> [low, high]
        mi, ma = score.min(1).values.unsqueeze(1), score.max(1).values.unsqueeze(1)
        S = (ma-mi) / (high-low)
        Z = high - ma / S
        if rounding:
            Z = torch.round(Z)
        q = score/S + Z
        if rounding:
            q = torch.round(q)
        return q


if __name__ == '__main__':
    import torch.optim as optim
    N = 6
    gt = torch.randint(0, N, size=(2,))
    scores = torch.rand(2, N)
    pred = torch.rand(2, N)
    pred.requires_grad = True
    ori = pred.clone()
    kd_loss = LambdaLoss(n_pos=3, n_neg=3, weight_scheme='ndcg2++', sigma=1., mu=1.)
    optimizer = optim.Adam([pred], lr=0.01)
    for epoch in range(10000):
        optimizer.zero_grad()
        loss = kd_loss(gt, scores, pred)
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print(f"{epoch} | loss: {loss.item()}")
    kd_loss(gt, scores, pred*100)
    print("original:", ori.sort(1, descending=True)[1].tolist())
    print("ground-truth:", scores.sort(1, descending=True)[1].tolist())
    print("updated:", pred.sort(1, descending=True)[1].tolist())