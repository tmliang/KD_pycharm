import torch
import torch.nn as nn
import torch.nn.functional as F


class ListNet(nn.Module):
    """
    Zhe Cao, Tao Qin, Tie-Yan Liu, Ming-Feng Tsai, and Hang Li. 2007.
    Learning to Rank: From Pairwise Approach to Listwise Approach. In Proceedings of the 24th ICML. 129–136.
    """
    def __init__(self, temperature=1):
        super().__init__()
        self.temperature = temperature

    def forward(self, score, tgt_score):
        """
        The Top-1 approximated ListNet kd_loss, which reduces to a softmax and simple cross entropy.
        """
        return F.kl_div(
            torch.log_softmax(score / self.temperature, dim=1),
            torch.softmax(tgt_score / self.temperature, dim=1),
            reduction="batchmean"
        ) * self.temperature ** 2

        # return -torch.sum(tgt_score.softmax(dim=1) * score.log_softmax(dim=1), dim=1).mean()

#
# if __name__ == '__main__':
# 	gt = torch.randint(0, 10, size=(4,))
# 	scores = torch.randn(4, 10)
# 	pred = nn.Parameter(torch.randn(4, 10))
# 	kd_loss = ListNet(n_pos=2, n_neg=5)
# 	l = kd_loss(gt, scores, pred)
# 	l.backward()
# 	print(l)
