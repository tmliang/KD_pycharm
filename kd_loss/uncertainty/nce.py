"""
Refer to https://github.com/naver-ai/pcme/blob/main/criterions/probemb.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def batchwise_cdist(samples1, samples2):
    """Compute L2 distance between each pair of the two multi-head embeddings in batch-wise.
    We may assume that samples have shape N x K x D, N: batch_size, K: number of embeddings, D: dimension of embeddings.
    The size of samples1 and samples2 (`N`) should be either
    - same (each sample-wise distance will be computed separately)
    - len(samples1) = 1 (samples1 will be broadcasted into samples2)
    - len(samples2) = 1 (samples2 will be broadcasted into samples1)

    The following broadcasting operation will be computed:
    (N x 1 x K x D) - (N x K x 1 x D) = (N x K x K x D)

    Parameters
    ----------
    samples1: torch.Tensor (shape: N x K x D)
    samples2: torch.Tensor (shape: N x K x D)

    Returns
    -------
    batchwise distance: N x K ** 2
    """
    if len(samples1.size()) != 3 or len(samples2.size()) != 3:
        raise RuntimeError('expected: 3-dim tensors, got: {}, {}'.format(samples1.size(), samples2.size()))

    if samples1.size(0) == samples2.size(0):
        batch_size = samples1.size(0)
    elif samples1.size(0) == 1:
        batch_size = samples2.size(0)
    elif samples2.size(0) == 1:
        batch_size = samples1.size(0)
    else:
        raise RuntimeError(f'samples1 ({samples1.size()}) and samples2 ({samples2.size()}) dimensionalities '
                           'are non-broadcastable.')

    samples1 = samples1.unsqueeze(1)
    samples2 = samples2.unsqueeze(2)
    return torch.norm(samples1 - samples2, dim=-1).view(batch_size, -1)


def soft_contrastive_nll(logit, matched):
    r"""Compute the negative log-likelihood of the soft contrastive loss.

    .. math::
        NLL_{ij} = -\log p(m = m_{ij} | z_i, z_j)
                 = -\log \left[ \mathbb{I}_{m_{ij} = 1} \sigma(-a \| z_i - z_j \|_2 + b)
                         +  \mathbb{I}_{m_{ij} = -1} (1 - \sigma(-a \| z_i - z_j \|_2 + b)) \right].

    Note that the matching indicator {m_ij} is 1 if i and j are matched otherwise -1.
    Here we define the sigmoid function as the following:
    .. math::
        \sigma(x) = \frac{\exp(x)}{\exp(x) + \exp(-x)}, \text{ i.e., }
        1 - \sigma(x) = \frac{\exp(-x)}{\exp(x) + \exp(-x)}.

    Here we sample "logit", s_{ij} by Monte-Carlo sampling to get the expected soft contrastive loss.
    .. math::
        s_{ij}^k = -a \| z_i^k - z_j^k \|_2 + b, z_i^k ~ \mathcal N (\mu_i, \Sigma_i), z_j^k ~ \mathcal N (\mu_j, \Sigma_j).

    Then we can compute NLL by logsumexp (here, we omit `k` in s_{ij}^k for the simplicity):
    .. math::
        NLL_{ij} = -\log \left[ \frac{1}{K^2} \sum_{s_{ij}} \left{ \frac{\exp(s_{ij} m_ij)}{\exp(s_{ij}) + \exp(-s_{ij})} \right} \right]
                 = (\log K^2) -\log \sum_{s_{ij}} \left[ \exp \left( s_{ij} m_ij - \log(\exp(s_{ij} + (-s_{ij}))) \right) \right]
                 = (\log K^2) -logsumexp( s_{ij} m_{ij} - logsumexp(s_{ij}, -s_{ij}) ).

    Parameters
    ----------
    logit: torch.Tensor (shape: N x K ** 2)
    matched: torch.Tensor (shape: N), an element should be either 1 (matched) or -1 (mismatched)

    Returns
    -------
    NLL loss: torch.Tensor (shape: N), should apply `reduction` operator for the backward operation.
    """
    if len(matched.size()) == 1:
        matched = matched[:, None]
    return -(
        (logit * matched - torch.stack(
            (logit, -logit), dim=2).logsumexp(dim=2, keepdim=False)
         ).logsumexp(dim=1)) + np.log(logit.size(1))


class NceLoss(nn.Module):
    def __init__(self, scale=0.0005, bias=0):
        super().__init__()
        self.scale = nn.Parameter(scale * torch.ones(1))
        self.bias = nn.Parameter(bias * torch.ones(1))

    def pairwise_sampling(self, anchors, candidates):
        N = len(anchors)
        if len(anchors) != len(candidates):
            raise RuntimeError('# anchors ({}) != # candidates ({})'.format(anchors.shape, candidates.shape))
        anchor_idx, selected_idx, matched = self.full_sampling(N)

        anchors = anchors[anchor_idx]
        selected = candidates[selected_idx]

        cdist = batchwise_cdist(anchors, selected)

        return cdist, matched

    def full_sampling(self, N):
        candidates = []
        selected = []
        matched = []
        for i in range(N):
            for j in range(N):
                candidates.append(i)
                selected.append(j)
                if i == j:
                    matched.append(1)
                else:
                    matched.append(-1)
        return candidates, selected, matched

    def forward(self, t_samples, s_samples):
        """
        Shape
        -----
        Input1 : torch.Tensor
            :math:`(N, K, D)` shape, `N` is the batch size, `K` is the number of samples and `D` is the size of a sample
        Input2: torch.Tensor
            :math:`(N, K, D)` shape, `N` is the batch size, `K` is the number of samples and `D` is the size of a sample
        Output: torch.Tensor
            If :attr:`reduction` is ``'none'``, then :math:`(N)`.
        """
        distance, matched = self.pairwise_sampling(t_samples, s_samples)
        matched = torch.tensor(matched, dtype=torch.long, device=distance.device)
        logits = -self.scale * distance + self.bias

        idx = matched == 1
        loss_pos = soft_contrastive_nll(logits[idx], matched[idx]).sum()
        idx = matched != 1
        loss_neg = soft_contrastive_nll(logits[idx], matched[idx]).sum()

        return (loss_pos + loss_neg) / 2
