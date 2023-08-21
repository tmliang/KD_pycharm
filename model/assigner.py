import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv


class WeightAssigner(nn.Module):
    def __init__(self, num_layer=3, num_edge=8, ds_node=2, ds_edge=2, hidden_size=64, dropout=0.1):
        super().__init__()
        self.num_layer = num_layer
        self.num_edge = num_edge
        self.hidden_size = hidden_size
        self.ds_node = ds_node
        self.ds_edge = ds_edge
        self.fc1 = nn.Linear(4, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.gnns = nn.ModuleList([GCNConv(hidden_size, hidden_size) for _ in range(num_layer)])
        self.act = nn.LeakyReLU()
        self.ln = nn.LayerNorm(4)
        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x, edge_score, node_score):
        graphs, selected_nodes, selected_nodes_iter = self.build_graphs(edge_score, node_score)
        x = self.get_stat_feat(x)
        x = self.fc1(x)
        for i, layer in enumerate(self.gnns):
            if i > 0:
                x = x.gather(1, selected_nodes_iter[i].unsqueeze(2).repeat(1, 1, self.hidden_size))
            x = layer(x.view(-1, self.hidden_size), graphs[i].edge_index).view(x.size())
            x = self.drop(self.act(x))
        x = self.fc2(x)
        x = torch.sigmoid(x @ x.transpose(1, 2))
        return x, selected_nodes

    def build_graphs(self, edge_score, node_score):
        graphs = []
        b, n = node_score.size()
        k = self.num_edge
        device = edge_score.device
        selected_nodes = torch.arange(n, device=device)[None, :].repeat(b, 1)
        selected_nodes_iter = [None]

        for s in range(self.num_layer):
            edge_softmax = torch.softmax((-edge_score.masked_fill(torch.eye(n, device=device).bool(), float('inf'))), -1)
            node_softmax = torch.softmax(-node_score, -1)

            # sample edges
            indices = torch.multinomial(edge_softmax.view(-1, n), k).view(b, n, -1).contiguous()
            head = indices.view(b, -1)
            tail = torch.arange(n, device=device).repeat_interleave(k).unsqueeze(0).repeat(b, 1)
            edge_index = torch.stack([head, tail], dim=1)  # directed graph
            graphs.append(Batch.from_data_list([Data(edge_index=edge_index[i], num_nodes=n) for i in range(b)]))
            k = k // self.ds_edge

            # sample nodes
            if s < self.num_layer - 1:
                n = n // self.ds_node
                indices = torch.multinomial(node_softmax, n)

                selected_nodes = selected_nodes.gather(1, indices)
                selected_nodes_iter.append(indices)

                # update graphs
                edge_score = edge_score.gather(1, indices.unsqueeze(2).expand(-1, -1, node_score.size(1))).gather(
                    2, indices.unsqueeze(1).expand(-1, n, -1))
                node_score = node_score.gather(1, indices)

        return graphs, selected_nodes, selected_nodes_iter

    def get_stat_feat(self, x, eps=1e-10):
        x1 = x.mean(1)
        x2 = 1 / (x.std(1) + eps)
        x_sort = x.argsort().argsort().float()
        x3 = x_sort.mean(1)
        x4 = 1 / (x_sort.std(1) + eps)
        x = torch.stack([x1, x2, x3, x4], dim=-1)
        x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + eps)
        return x
