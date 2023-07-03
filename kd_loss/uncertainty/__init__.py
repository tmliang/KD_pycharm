import torch


def kl_loss(mu, sigma):
    return -0.5 * (1 + sigma.log() - mu.pow(2) - sigma).sum(-1).mean()


def uniform_loss(mu, t=2):
    return torch.pdist(mu).pow(2).mul(-t).exp().mean()
