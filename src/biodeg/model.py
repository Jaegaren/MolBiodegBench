import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(torch.nn.Module):



    def __init__(self, in_channels=7, hidden=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.bn1 = torch.nn.BatchNorm1d(hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.bn2 = torch.nn.BatchNorm1d(hidden)
        self.dropout = torch.nn.Dropout(0.3)
        self.head = torch.nn.Linear(hidden, 2)


    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        return self.head(x)