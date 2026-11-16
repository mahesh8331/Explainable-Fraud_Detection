import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

HUB_CAP = 5000
DROP = ["TransactionID", "TransactionDT", "isFraud", "split"]


def build_graph(splits):
    """Build the graph from the rows whose split is in `splits`.
    Cleaning statistics always come from the training rows only."""
    df = pd.read_parquet("data/processed/clean.parquet")
    d = df[df["split"].isin(splits)].reset_index(drop=True)
    is_train = (d["split"] == "train").values

    # ---- Transaction features (statistics from training rows only) ----
    feature_cols = [c for c in d.columns if c not in DROP]
    X = d[feature_cols].copy()
    X["TransactionAmt"] = np.log1p(X["TransactionAmt"])

    miss_share = X[is_train].isna().mean()
    miss_cols = miss_share[miss_share > 0.10].index
    indicators = X[miss_cols].isna().astype("float32").add_suffix("_missing")

    X = X.fillna(X[is_train].median())
    means = X[is_train].mean()
    stds = X[is_train].std().replace(0, 1)
    X = (X - means) / stds
    X = pd.concat([X, indicators], axis=1).astype("float32")

    # ---- Card nodes ----
    card_key = d["card1"].astype(str) + "_" + d["card2"].fillna(-1).astype(str)
    card_idx, card_names = pd.factorize(card_key)

    # ---- Device nodes (hub values judged on training rows only) ----
    dev = d["DeviceInfo"]
    counts = dev[is_train].value_counts()
    too_common = counts[counts > HUB_CAP].index
    dev = dev.where(~dev.isin(too_common))
    dev_idx, dev_names = pd.factorize(dev)

    # ---- Edges ----
    tx_ids = np.arange(len(d))
    has_dev = dev_idx != -1

    # ---- Pack everything ----
    data = HeteroData()
    data["transaction"].x = torch.tensor(X.values)
    data["transaction"].y = torch.tensor(d["isFraud"].values, dtype=torch.long)
    data["transaction"].train_mask = torch.tensor(is_train)
    data["transaction"].val_mask = torch.tensor((d["split"] == "val").values)
    data["card"].num_nodes = len(card_names)
    data["device"].num_nodes = len(dev_names)
    data["transaction", "uses", "card"].edge_index = torch.tensor(
        np.stack([tx_ids, card_idx]), dtype=torch.long)
    data["transaction", "uses", "device"].edge_index = torch.tensor(
        np.stack([tx_ids[has_dev], dev_idx[has_dev]]), dtype=torch.long)
    return T.ToUndirected()(data)


if __name__ == "__main__":
    for splits in (["train"], ["train", "val"]):
        g = build_graph(splits)
        tx = g["transaction"]
        print(splits, "-> features:", tuple(tx.x.shape),
              "| train rows:", int(tx.train_mask.sum()),
              "| val rows:", int(tx.val_mask.sum()),
              "| cards:", g["card"].num_nodes,
              "| devices:", g["device"].num_nodes)
