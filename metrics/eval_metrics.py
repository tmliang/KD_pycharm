import torch


def mean_average_precision(rank, label):
    res = torch.nonzero(rank == label.unsqueeze(1))[:, 1] + 1
    return torch.sum(1/res) / label.size(0)


if __name__ == '__main__':
    label = torch.tensor([2, 1])
    pred = torch.randn(2, 5)
    rank = torch.topk(pred, k=2).indices
    print(f"label:{label} \n rank:{rank} \n map:{mean_average_precision(rank, label)}")
