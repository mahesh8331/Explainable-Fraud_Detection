import time
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from graph_data import build_graph
from gnn_model import FraudGNN

EPOCHS = 150         # short test run first; raise to 150 for the real run
EVAL_EVERY = 5    # check the validation score every this many epochs
PATIENCE = 6        # stop if no improvement after this many checks
LR = 0.01

USE_GRAPH = False   # False = control experiment (graph switched off)
TAG = "gnn" if USE_GRAPH else "gnn_nograph"

torch.manual_seed(42)

print("Building graphs (this takes a minute or two)...")
g_train = build_graph(["train"])
g_eval = build_graph(["train", "val"])
print("train graph:", tuple(g_train["transaction"].x.shape),
      "| eval graph:", tuple(g_eval["transaction"].x.shape))

y_train = g_train["transaction"].y
pos_weight = ((y_train == 0).sum() / (y_train == 1).sum()).float()
print("Fraud weight:", round(float(pos_weight), 1))

model = FraudGNN(in_dim=g_train["transaction"].x.shape[1], use_graph=USE_GRAPH)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

val_mask = g_eval["transaction"].val_mask
y_val = g_eval["transaction"].y[val_mask].numpy()


def val_scores():
    model.eval()
    with torch.no_grad():
        return model(g_eval)[val_mask].numpy()


def recall_at_precision(y_true, scores, min_precision):
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ok = precision[:-1] >= min_precision
    return recall[:-1][ok].max() if ok.any() else 0.0


best, checks_without_gain = 0.0, 0
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()
    optimizer.zero_grad()
    logits = model(g_train)
    loss = F.binary_cross_entropy_with_logits(logits, y_train.float(), pos_weight=pos_weight)
    loss.backward()
    optimizer.step()
    msg = f"epoch {epoch:3d} | loss {loss.item():.4f}"

    if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
        pr = average_precision_score(y_val, val_scores())
        msg += f" | val PR-AUC {pr:.4f}"
        if pr > best:
            best, checks_without_gain = pr, 0
            torch.save(model.state_dict(), f"models/{TAG}_best.pt")

        else:
            checks_without_gain += 1
    print(msg + f" | {time.time() - t0:.1f}s", flush=True)
    if checks_without_gain >= PATIENCE:
        print("No improvement for a while, stopping early.")
        break

model.load_state_dict(torch.load(f"models/{TAG}_best.pt", weights_only=True))
scores = val_scores()
result = pd.DataFrame([{
    "model": "GNN (card + device graph)" if USE_GRAPH else "Same model, graph switched off",
    "PR-AUC": average_precision_score(y_val, scores),
    "ROC-AUC": roc_auc_score(y_val, scores),
    "recall@prec50": recall_at_precision(y_val, scores, 0.50),
    "recall@prec80": recall_at_precision(y_val, scores, 0.80),
}]).round(4)
result.to_csv(f"results/{TAG}_results.csv", index=False)
print("\n", result.to_string(index=False))
