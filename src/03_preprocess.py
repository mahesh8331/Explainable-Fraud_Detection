import os
import numpy as np
import pandas as pd

trans = pd.read_csv("data/raw/train_transaction.csv")
ident = pd.read_csv("data/raw/train_identity.csv")
df = trans.merge(ident, on="TransactionID", how="left").copy()

# 1. Drop columns that are almost entirely empty (>95% missing)
missing = df.isna().mean()
drop_cols = missing[missing > 0.95].index.tolist()
df = df.drop(columns=drop_cols)
print("Dropped", len(drop_cols), "columns:", drop_cols)

# 2. Turn text columns into numbers (a number per category; NaN stays NaN)
text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
for c in text_cols:
    codes, _ = pd.factorize(df[c])
    df[c] = np.where(codes == -1, np.nan, codes).astype("float32")
print("Converted", len(text_cols), "text columns to numbers")

# 3. Add the time-based split (70% train / 15% val / 15% test)
t1 = df["TransactionDT"].quantile(0.70)
t2 = df["TransactionDT"].quantile(0.85)
df["split"] = np.where(df["TransactionDT"] <= t1, "train",
                np.where(df["TransactionDT"] <= t2, "val", "test"))

# 4. Save in a fast format
os.makedirs("data/processed", exist_ok=True)
df.to_parquet("data/processed/clean.parquet", index=False)

print("\nFinal shape:", df.shape)
print(df.groupby("split")["isFraud"].agg(["count", "mean"]))
