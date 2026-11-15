import pandas as pd

df = pd.read_parquet("data/processed/clean.parquet")
train = df[df["split"] == "train"].reset_index(drop=True)   # training rows only

# Transaction nodes: the row position 0 ... 413377 is the node ID
print("Transaction nodes:", len(train))

# Card nodes: one per unique card1 + card2 combination
card_key = train["card1"].astype(str) + "_" + train["card2"].fillna(-1).astype(str)
card_idx, card_names = pd.factorize(card_key)       # card_idx: card node ID for each transaction
print("Card nodes:", len(card_names))

# Device nodes: ignore values shared by too many transactions (generic hubs)
HUB_CAP = 5000
dev = train["DeviceInfo"]
counts = dev.value_counts()
too_common = counts[counts > HUB_CAP].index
print("Device values removed as hubs:", len(too_common))
dev = dev.where(~dev.isin(too_common))              # removed values become empty = no edge
dev_idx, dev_names = pd.factorize(dev)              # -1 means "no device edge"
print("Device nodes:", len(dev_names))
print("Transactions with a device edge:", (dev_idx != -1).sum())

import numpy as np

tx_ids = np.arange(len(train))                      # transaction node IDs: 0, 1, 2, ...

# Transaction -> card edges: every transaction has a card
tx_card_edges = np.stack([tx_ids, card_idx])
print("\nTransaction-card edges:", tx_card_edges.shape[1])

# Transaction -> device edges: only where a device node exists (-1 means none)
has_dev = dev_idx != -1
tx_dev_edges = np.stack([tx_ids[has_dev], dev_idx[has_dev]])
print("Transaction-device edges:", tx_dev_edges.shape[1])


# ---- Transaction features ----
drop = ["TransactionID", "TransactionDT", "isFraud", "split"]
feature_cols = [c for c in train.columns if c not in drop]
X = train[feature_cols].copy()
y = train["isFraud"].values

# 1. Log of the amount
X["TransactionAmt"] = np.log1p(X["TransactionAmt"])

# 2. "Was missing" markers for columns that are more than 10% empty
miss_share = X.isna().mean()
miss_cols = miss_share[miss_share > 0.10].index
indicators = X[miss_cols].isna().astype("float32").add_suffix("_missing")

# 3. Fill empty cells with the training median
medians = X.median()
X = X.fillna(medians)

# 4. Standardise: (value - average) / spread, using training statistics
means = X.mean()
stds = X.std().replace(0, 1)          # avoid dividing by zero for constant columns
X = (X - means) / stds

# 5. Put the pieces together
X = pd.concat([X, indicators], axis=1).astype("float32")

print("\nFeature table shape:", X.shape)
print("Columns with 'was missing' marker:", len(miss_cols))
print("Empty cells left:", int(X.isna().sum().sum()))
print("Fraud labels:", int(y.sum()), "of", len(y))


# ---- Transaction features ----
drop = ["TransactionID", "TransactionDT", "isFraud", "split"]
feature_cols = [c for c in train.columns if c not in drop]
X = train[feature_cols].copy()
y = train["isFraud"].values

# 1. Log of the amount
X["TransactionAmt"] = np.log1p(X["TransactionAmt"])

# 2. "Was missing" markers for columns that are more than 10% empty
miss_share = X.isna().mean()
miss_cols = miss_share[miss_share > 0.10].index
indicators = X[miss_cols].isna().astype("float32").add_suffix("_missing")

# 3. Fill empty cells with the training median
medians = X.median()
X = X.fillna(medians)

# 4. Standardise: (value - average) / spread, using training statistics
means = X.mean()
stds = X.std().replace(0, 1)          # avoid dividing by zero for constant columns
X = (X - means) / stds

# 5. Put the pieces together
X = pd.concat([X, indicators], axis=1).astype("float32")

print("\nFeature table shape:", X.shape)
print("Columns with 'was missing' marker:", len(miss_cols))
print("Empty cells left:", int(X.isna().sum().sum()))
print("Fraud labels:", int(y.sum()), "of", len(y))


import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

data = HeteroData()

# Nodes
data["transaction"].x = torch.tensor(X.values)                  # features
data["transaction"].y = torch.tensor(y, dtype=torch.long)       # fraud labels
data["card"].num_nodes = len(card_names)
data["device"].num_nodes = len(dev_names)

# Edges
data["transaction", "uses", "card"].edge_index = torch.tensor(tx_card_edges, dtype=torch.long)
data["transaction", "uses", "device"].edge_index = torch.tensor(tx_dev_edges, dtype=torch.long)

# Add the reverse direction (card -> transaction, device -> transaction)
data = T.ToUndirected()(data)
print("\n", data)

# Sanity checks: every edge must point at a node that exists
e1 = data["transaction", "uses", "card"].edge_index
e2 = data["transaction", "uses", "device"].edge_index
assert e1[0].max() < len(train) and e1[1].max() < len(card_names)
assert e2[0].max() < len(train) and e2[1].max() < len(dev_names)
print("Edge checks passed")
