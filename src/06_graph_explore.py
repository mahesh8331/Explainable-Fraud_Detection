import pandas as pd

df = pd.read_parquet("data/processed/clean.parquet")
train = df[df["split"] == "train"]          # graph uses training rows only
print("Training transactions:", len(train))

# Candidate definitions of "one card"
keys = {
    "card1 alone": train["card1"].astype(str),
    "card1 + card2": train["card1"].astype(str) + "_" + train["card2"].fillna(-1).astype(str),
    "card1 + card2 + addr1": (train["card1"].astype(str) + "_"
                              + train["card2"].fillna(-1).astype(str) + "_"
                              + train["addr1"].fillna(-1).astype(str)),
    # the other two node types, for comparison
    "DeviceInfo": train["DeviceInfo"],
    "addr1": train["addr1"],
}

for name, key in keys.items():
    vc = key.value_counts()                  # how many transactions per value
    print(f"\n{name}")
    print("  missing:           ", f"{key.isna().mean():.1%}")
    print("  unique values:     ", len(vc))
    print("  shared by 2+ txns: ", f"{(vc > 1).mean():.1%} of values")
    print("  biggest group:     ", vc.max(), "transactions")

raw = pd.read_csv("data/raw/train_identity.csv")
print(raw["DeviceInfo"].value_counts().head(8))
