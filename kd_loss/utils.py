import torch


def pair_minus(x):
    """
    :param x: [*, N]
    :return: [*, N, N]
    """
    # A[i, j] = x[i] - x[j]
    return x.unsqueeze(-1) - x.unsqueeze(-2)


def Sinkhorn(cost, max_iter=100, temperature=1, thresh=1e-8):
    bsz, m, n = cost.size()
    cost = torch.exp(-cost / temperature)
    u = torch.zeros(bsz, m, 1).to(cost).fill_(1. / m)
    v = torch.zeros(bsz, n, 1).to(cost).fill_(1. / n)

    ut = torch.ones_like(u)
    vt = torch.ones_like(v)
    for t in range(max_iter):
        ul = ut
        ut = u / torch.matmul(cost, vt)
        vt = v / torch.matmul(cost.transpose(1, 2).contiguous(), ut)
        err = (ut - ul).abs().mean()
        if err.item() < thresh:
            break

    T = torch.matmul(ut, vt.transpose(1, 2)) * cost     # scale with num_ans

    return T