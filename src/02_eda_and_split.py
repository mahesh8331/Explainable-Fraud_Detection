import pandas as pd

trans = pd.read_csv("data/raw/train_transaction.csv")
ident = pd.read_csv("data/raw/train_identity.csv")
df = trans.merge(ident, on="TransactionID", how="left")   # attach identity info where it exists

# 1. Fraud rate over time (30-day blocks)
df["day"] = df["TransactionDT"] // 86400
df["month_block"] = df["day"] // 30
print("Fraud rate by 30-day block:")
print(df.groupby("month_block")["isFraud"].agg(["count", "mean"]))

# 2. Which columns have the most missing values?
missing = df.isna().mean().sort_values(ascending=False)
print("\nTop 10 most-missing columns:")
print(missing.head(10))
print("Columns with more than 50% missing:", (missing > 0.5).sum(), "of", df.shape[1])

# 3. Candidate graph 'entities': how many unique values, and how much are they reused?
entities = ["card1", "addr1", "P_emaildomain", "R_emaildomain", "DeviceInfo", "ProductCD"]
print("\nEntity columns:")
for col in entities:
    vc = df[col].value_counts()
    print(f"{col:15s} unique={df[col].nunique():>7}  "
          f"missing={df[col].isna().mean():.1%}  "
          f"shared by 2+ transactions={(vc > 1).mean():.1%} of values")

# 4. Proposed time split: 70% / 15% / 15% by time order
t1 = df["TransactionDT"].quantile(0.70)
t2 = df["TransactionDT"].quantile(0.85)
split = pd.cut(df["TransactionDT"], [-1, t1, t2, df["TransactionDT"].max()],
               labels=["train", "val", "test"])
print("\nTime split:")
print(df.groupby(split, observed=True)["isFraud"].agg(["count", "mean"]))
print("Days: train ends", round(t1 / 86400, 1), "| val ends", round(t2 / 86400, 1))
