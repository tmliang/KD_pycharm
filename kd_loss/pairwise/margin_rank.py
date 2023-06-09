import torch
import torch.nn as nn
from itertools import combinations


class MarginRank(nn.Module):
    """
    LED: Lexicon-Enlightened Dense Retriever for Large-Scale Retrieval. WWW. 2023
    """
    def __init__(self, n_pos, margin=0):
        super().__init__()
        self.n_pos = n_pos
        self.margin = margin
        self.margin_ranking_loss = nn.MarginRankingLoss(margin=margin, reduction='mean')

    def forward(self, tgt_score, score):
        # get pairwise indices
        bsz, list_len = score.size()
        target_num = int(self.n_pos * (list_len - (1 + self.n_pos) / 2))    # ignore the neg-neg pairs
        pair_ind = torch.tensor(list(combinations(torch.arange(list_len), 2))[:target_num],
                                device=score.device, dtype=torch.long)
        si = score.gather(1, pair_ind[:, 0].expand(bsz, -1))
        sj = score.gather(1, pair_ind[:, 1].expand(bsz, -1))
        return self.margin_ranking_loss(si, sj, torch.ones_like(si))


if __name__ == '__main__':
    import torch.optim as optim
    N = 10
    gt = torch.randint(0, N, size=(2,))
    scores = torch.rand(2, N) * 100
    pred = torch.rand(2, N) * 100
    pred.requires_grad = True
    ori = pred.clone()
    kd_loss = MarginRank(n_pos=3, n_neg=3, neg_sampler='random')
    optimizer = optim.Adam([pred], lr=0.01)
    for epoch in range(10000):
        optimizer.zero_grad()
        loss = kd_loss(gt.cuda(), scores.cuda(), pred.cuda())
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print(f"{epoch} | loss: {loss.item()}")
    kd_loss(gt, scores, pred*100)
    print("original:", ori.sort(1, descending=True)[1].tolist())
    print("ground-truth:", scores.sort(1, descending=True)[1].tolist())
    print("updated:", pred.sort(1, descending=True)[1].tolist())
