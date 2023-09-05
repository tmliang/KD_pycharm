import torch


def ndcg_at_k(pred, label, k=5):

    def dcg_at_k(score):
        G = 2 ** score - 1  # gains
        D = (1 / torch.log2(torch.arange(2, G.size(1) + 2, device=score.device))).unsqueeze(0)  # discounts
        return torch.sum(G * D, dim=1)

    pred_score = label.gather(1, torch.topk(pred, k=k).indices)
    ideal_score = torch.topk(label, k=k).values
    zero_id = ideal_score[:, 0] == 0
    dcg = dcg_at_k(pred_score)
    idcg = dcg_at_k(ideal_score)
    ndcg = dcg / idcg
    ndcg[zero_id] = 0
    return ndcg


# Test
if __name__ == "__main__":
    # Sample data
    pred = torch.tensor([[0.2, 0.4, 0.1, 0.3], [0.5, 0.3, 0.1, 0.1]])
    label = torch.tensor([[3, 2, 1, 0], [1, 0, 2, 3]])

    # Call ndcg_at_k
    result = ndcg_at_k(pred, label, k=3)

    # Print the result
    print(result)
