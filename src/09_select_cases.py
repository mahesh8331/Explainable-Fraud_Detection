import numpy as np
import pandas as pd
import torch

from graph_data import build_graph
from gnn_model import FraudGNN

N_CASES = 40

g = build_graph(["train", "val"])
tx = g["transaction"]
print("number columns:", len(g.feature_names), "| category columns:", len(g.cat_names))

model = FraudGNN(in_dim=tx.x.shape[1], cat_sizes=tx.cat_sizes.tolist(), use_graph=True)
model.load_state_dict(torch.load("models/gnn_emb_best.pt", weights_only=True))
model.eval()
with torch.no_grad():
    scores = model(g)

val = tx.val_mask.nonzero().flatten()
cut = torch.quantile(scores[val], 0.99)
flagged = val[scores[val] >= cut]                  # the top 1% highest scores in validation
hit = flagged[tx.y[flagged] == 1]                  # of those, the real frauds
print(f"flagged (top 1%): {len(flagged)} | real frauds among them: {len(hit)} "
      f"| precision {len(hit) / len(flagged):.1%}")

rng = np.random.default_rng(42)
pick = rng.choice(hit.numpy(), size=min(N_CASES, len(hit)), replace=False)
cases = pd.DataFrame({
    "node": pick,
    "TransactionID": tx.tx_id[pick].numpy(),
    "score": scores[pick].numpy().round(3),
}).sort_values("score", ascending=False)
cases.to_csv("results/cases.csv", index=False)
print(cases.head(5).to_string(index=False))
print("...", len(cases), "cases saved")
