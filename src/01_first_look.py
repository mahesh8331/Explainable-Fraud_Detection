import pandas as pd

# 1. Load the two training tables
trans = pd.read_csv("data/raw/train_transaction.csv")
ident = pd.read_csv("data/raw/train_identity.csv")

# 2. How big are they? (rows, columns)
print("transactions:", trans.shape)
print("identity:", ident.shape)

# 3. What does the data look like?
print(trans.head(3).T.head(15))

# 4. How much is fraud?
print("\nFraud rate:", trans["isFraud"].mean())
print(trans["isFraud"].value_counts())

# 5. What time range does the data cover? (seconds from a start point)
print("\nTransactionDT min/max:", trans["TransactionDT"].min(), trans["TransactionDT"].max())
print("Days covered:", (trans["TransactionDT"].max() - trans["TransactionDT"].min()) / 86400)

# 6. How many transactions have identity info?
has_id = trans["TransactionID"].isin(ident["TransactionID"])
print("\nShare of transactions with identity info:", has_id.mean())
