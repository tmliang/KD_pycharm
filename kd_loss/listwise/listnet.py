import torch
import torch.nn.functional as F
from kd_loss.base import BaseLoss


class ListNet(BaseLoss):
    """
    Zhe Cao, Tao Qin, Tie-Yan Liu, Ming-Feng Tsai, and Hang Li. 2007.
    Learning to Rank: From Pairwise Approach to Listwise Approach. In Proceedings of the 24th ICML. 129–136.
    """
    def __init__(self, n_pos=10, n_neg=50, neg_sampler='prob', s_neg=False, temperature=1):
        super().__init__(n_pos, n_neg, neg_sampler, s_neg)
        self.temperature = temperature

    def forward(self, gt, t_score, s_score):
        """
        The Top-1 approximated ListNet kd_loss, which reduces to a softmax and simple cross entropy.
        """
        t_score, s_score = self.sort_scores_by_teacher(gt, t_score, s_score)
        return F.kl_div(
            torch.log_softmax(s_score / self.temperature, dim=1),
            torch.softmax(t_score / self.temperature, dim=1),
            reduction="batchmean"
        ) * self.temperature ** 2

        # return -torch.sum(t_score.softmax(dim=1) * s_score.log_softmax(dim=1), dim=1).mean()

#
# if __name__ == '__main__':
# 	gt = torch.randint(0, 10, size=(4,))
# 	scores = torch.randn(4, 10)
# 	pred = nn.Parameter(torch.randn(4, 10))
# 	kd_loss = ListNet(n_pos=2, n_neg=5)
# 	l = kd_loss(gt, scores, pred)
# 	l.backward()
# 	print(l)
