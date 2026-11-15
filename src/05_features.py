import numpy as np
import pandas as pd

df = pd.read_parquet("data/processed/clean.parquet")

# 1. Time features. TransactionDT is seconds from an unknown start, so these are
#    "relative" hour and weekday: the repeating daily/weekly pattern is still visible.
df["hour"] = (df["TransactionDT"] // 3600) % 24
df["dayofweek"] = (df["TransactionDT"] // 86400) % 7

# 2. Amount: log squashes the huge skew (median 68, max 31,937)
df["amt_log"] = np.log1p(df["TransactionAmt"])

# 3. A better card identifier: card1 + card2 + addr1 together describe one card holder better
#    than any one column. The text is only used to give each combination a number.
key = (df["card1"].astype(str) + "_" + df["card2"].astype(str) + "_" + df["addr1"].astype(str))
df["card_id"] = pd.factorize(key)[0].astype("float32")

# 4. Frequency counts: how often has this card / address been seen?
#    Counted on TRAIN rows only, so validation and test never leak into the counts.
#    Values never seen in train get 0.
is_train = df["split"] == "train"
for col in ["card_id", "card1", "addr1"]:
    counts = df.loc[is_train, col].value_counts()
    df[col + "_count"] = df[col].map(counts).fillna(0).astype("float32")

new_cols = ["hour", "dayofweek", "amt_log", "card_id",
            "card_id_count", "card1_count", "addr1_count"]
print("Added", len(new_cols), "columns:", new_cols)
print(df[new_cols].describe().T[["mean", "min", "max"]])
print("Final shape:", df.shape)

df.to_parquet("data/processed/features.parquet", index=False)
