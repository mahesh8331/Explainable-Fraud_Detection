import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

HUB_CAP = 5000
MIN_COUNT = 50     # a category value seen fewer times than this in training becomes "other"
DROP = ["TransactionID", "TransactionDT", "isFraud", "split"]

# The 29 text columns we turned into integer codes in 03_preprocess.py
CAT_COLS = (["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain"]
            + [f"M{k}" for k in range(1, 10)]
            + ["id_12", "id_15", "id_16", "id_28", "id_29", "id_30", "id_31", "id_33",
               "id_34", "id_35", "id_36", "id_37", "id_38", "DeviceType", "DeviceInfo"])


def build_graph(splits):
    """Build the graph from the rows whose split is in `splits`.
    Cleaning statistics always come from the training rows only."""
    df = pd.read_parquet("data/processed/clean.parquet")
    d = df[df["split"].isin(splits)].reset_index(drop=True)
    is_train = (d["split"] == "train").values

    card_key = d["card1"].astype(str) + "_" + d["card2"].fillna(-1).astype(str)

    # ---- Category columns -> small whole-number IDs (0 = missing, 1 = other) ----
    d["hour"] = (d["TransactionDT"] // 3600) % 24
    d["dayofweek"] = (d["TransactionDT"] // 86400) % 7
    cat_cols = CAT_COLS + ["hour", "dayofweek"]
    cat_ids, cat_sizes = [], []
    for c in cat_cols:
        values = d[c]
        vc = values[is_train].value_counts()
        keep = vc[vc >= MIN_COUNT].index
        mapping = pd.Series(np.arange(len(keep)) + 2, index=keep)
        ids = values.map(mapping).fillna(1).where(values.notna(), 0)
        cat_ids.append(ids.astype("int64").values)
        cat_sizes.append(len(keep) + 2)

    # ---- Number columns (statistics from training rows only) ----
    num_cols = [c for c in d.columns if c not in DROP + cat_cols]
    X = d[num_cols].copy()
    X["TransactionAmt"] = np.log1p(X["TransactionAmt"])

    # How often has this card / address been seen in training?
    for name, key in [("card_count", card_key), ("card1_count", d["card1"]),
                      ("addr1_count", d["addr1"])]:
        counts = key[is_train].value_counts()
        X[name] = np.log1p(key.map(counts).fillna(0))

    miss_share = X[is_train].isna().mean()
    miss_cols = miss_share[miss_share > 0.10].index
    indicators = X[miss_cols].isna().astype("float32").add_suffix("_missing")

    X = X.fillna(X[is_train].median())
    means = X[is_train].mean()
    stds = X[is_train].std().replace(0, 1)
    X = (X - means) / stds
    X = pd.concat([X, indicators], axis=1).astype("float32")

    # ---- Card nodes ----
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
    data["transaction"].cat = torch.tensor(np.stack(cat_ids, axis=1), dtype=torch.long)
    data["transaction"].cat_sizes = torch.tensor(cat_sizes)
    data["transaction"].tx_id = torch.tensor(d["TransactionID"].values, dtype=torch.long)
    data["transaction"].y = torch.tensor(d["isFraud"].values, dtype=torch.long)
    data["transaction"].train_mask = torch.tensor(is_train)
    data["transaction"].val_mask = torch.tensor((d["split"] == "val").values)
    data.feature_names = list(X.columns)     # names of the number columns, in order
    data.cat_names = list(cat_cols)          # names of the category columns, in order
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
        print(splits, "-> numbers:", tuple(tx.x.shape), "| categories:", tuple(tx.cat.shape),
              "| train rows:", int(tx.train_mask.sum()),
              "| val rows:", int(tx.val_mask.sum()),
              "| cards:", g["card"].num_nodes,
              "| devices:", g["device"].num_nodes)
