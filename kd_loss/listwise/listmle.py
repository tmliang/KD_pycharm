import torch
from kd_loss.base import BaseLoss


class ListMLE(BaseLoss):
	"""
	ListMLE: Fen Xia, Tie-Yan Liu, Jue Wang, Wensheng Zhang, and Hang Li. 2008. Listwise Approach to Learning to Rank: Theory and Algorithm.
	In Proceedings of the 25th ICML. 1192–1199.
	"""
	def __init__(self, n_pos=10, n_neg=50, neg_sampler=None):
		super().__init__(n_pos, n_neg, neg_sampler)

	def forward(self, gt, t_score, s_score):
		"""
		ListMLE estimates the probability of the observed ranking and attempts to maximize it.
		"""
		t_score, s_score = self.sort_scores_by_teacher(gt, t_score, s_score)
		s_score = s_score - s_score.max(dim=1, keepdim=True).values.detach()		# keep exp() stable
		loss = torch.logcumsumexp(s_score.flip(1), dim=1).flip(1) - s_score
		return torch.mean(loss)


# if __name__ == '__main__':
# 	import torch.nn as nn
# 	gt = torch.randint(0, 10, size=(4,))
# 	scores = torch.randn(4, 10)
# 	pred = nn.Parameter(torch.randn(4, 10))
# 	kd_loss = ListMLE(n_pos=2, n_neg=5)
# 	l = kd_loss(gt, scores, pred)
# 	l.backward()
# 	print(l)