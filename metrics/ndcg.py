import torch


def dcg_at_k(rank):
    G = 2 ** rank - 1  # gains
    D = (1 / torch.log2(torch.arange(2, G.size(1) + 2, device=rank.device))).unsqueeze(0)  # discounts
    return torch.sum(G * D, dim=1, keepdim=True)


def ndcg_at_k(gt, label, pred, k=10):
    # ensure ground-truth labels to have max scores
    gt_score = (label.max(1).values + 1).unsqueeze(1)
    label = label.scatter(1, gt.unsqueeze(1), gt_score)

    # Quantization -> [0, 16]
    mi, ma = label.min(1).values.unsqueeze(1), label.max(1).values.unsqueeze(1)
    label = (label-mi) / (ma-mi+1e-10) * 16

    label_rank = torch.topk(label, k=k).values
    pred_rank = label.gather(1, torch.topk(pred, k=k).indices)
    idcg = dcg_at_k(label_rank)
    dcg = dcg_at_k(pred_rank)
    return torch.mean(dcg / idcg)



# if __name__ == '__main__':
#     label = torch.rand(2, 10)*100
#     pred = torch.rand(2, 10)*100
#     gt = torch.randint(low=0, high=10, size=(2,))
#     print(ndcg_at_k(gt, label, pred, k=5))