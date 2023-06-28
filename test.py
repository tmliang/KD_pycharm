import torch
import torch.distributions as tdist
import time


def f1(mu, logsigma, num_samples):
    eps = torch.randn(mu.size(0), num_samples, mu.size(1), dtype=mu.dtype, device=mu.device)

    samples = eps.mul(torch.exp(logsigma.unsqueeze(1))).add_(mu.unsqueeze(1))
    return samples


def f2(mu, logsigma, num_samples):
    eps = torch.randn(mu.size(0), num_samples, mu.size(1), dtype=mu.dtype, device=mu.device)

    samples = eps * torch.exp(logsigma.unsqueeze(1)) + mu.unsqueeze(1)
    return samples


if __name__ == '__main__':
    N = 10
    d = 1000
    n = 10000
    mu = torch.randn(N, d, device='cuda')
    logsigma = torch.randn(N, d, device='cuda')

    tic = time.time()
    f1(mu, logsigma, n)
    toc = time.time() - tic
    print(f"f1: {toc} s")

    tic = time.time()
    f2(mu, logsigma, n)
    toc = time.time() - tic
    print(f"f2: {toc} s")