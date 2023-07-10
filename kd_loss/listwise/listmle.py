import torch
import torch.nn as nn


class ListMLE(nn.Module):
	"""
	Listwise Approach to Learning to Rank: Theory and Algorithm. ICML 2008.
	"""
	def __init__(self):
		super().__init__()

	def forward(self, tgt_score, score):
		"""
		ListMLE estimates the probability of the observed ranking and attempts to maximize it.
		"""
		score = score - score.max(dim=1, keepdim=True).values.detach()		# keep exp() stable
		loss = torch.logcumsumexp(score.flip(1), dim=1).flip(1) - score
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