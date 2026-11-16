import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear


class FraudGNN(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.tx_in = Linear(in_dim, hidden)
        self.conv1 = HeteroConv({
            ("transaction", "uses", "card"): SAGEConv((hidden, hidden), hidden),
            ("transaction", "uses", "device"): SAGEConv((hidden, hidden), hidden),
        }, aggr="sum")
        self.conv2 = HeteroConv({
            ("card", "rev_uses", "transaction"): SAGEConv((hidden, hidden), hidden),
            ("device", "rev_uses", "transaction"): SAGEConv((hidden, hidden), hidden),
        }, aggr="sum")
        self.out = Linear(hidden * 2, 1)

    def forward(self, data):
        # Step 1: summarise each transaction
        h_tx = F.relu(self.tx_in(data["transaction"].x))

        # Cards and devices start empty (they have no notes of their own)
        start = {
            "transaction": h_tx,
            "card": torch.zeros(data["card"].num_nodes, h_tx.size(1)),
            "device": torch.zeros(data["device"].num_nodes, h_tx.size(1)),
        }

        # Step 2: cards and devices listen to their transactions
        mid = self.conv1(start, data.edge_index_dict)
        mid = {k: F.relu(v) for k, v in mid.items()}

        # Step 3: transactions listen to their cards and devices
        end = self.conv2({"transaction": h_tx, **mid}, data.edge_index_dict)
        heard = F.relu(end["transaction"])

        # Combine own summary + what was heard, and output one score per transaction
        return self.out(torch.cat([h_tx, heard], dim=1)).squeeze(-1)
