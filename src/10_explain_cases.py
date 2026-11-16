import json
import re
import time

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.explain import Explainer, GNNExplainer

from graph_data import build_graph
from gnn_model import FraudGNN

MAX_CASES = 40
TOP_K = 10        # how many features count as "the explanation"
EDGE_TYPES = [("transaction", "uses", "card"), ("card", "rev_uses", "transaction"),
              ("transaction", "uses", "device"), ("device", "rev_uses", "transaction")]
EMPTY = torch.empty((2, 0), dtype=torch.long)

torch.manual_seed(0)
g = build_graph(["train", "val"])
tx = g["transaction"]
names, cat_names = g.feature_names, g.cat_names

model = FraudGNN(in_dim=tx.x.shape[1], cat_sizes=tx.cat_sizes.tolist(), use_graph=True)
model.load_state_dict(torch.load("models/gnn_emb_best.pt", weights_only=True))
model.eval()
with torch.no_grad():
    scores = model(g)

card_of_tx = g[EDGE_TYPES[0]].edge_index[1]      # edges were built in transaction order
ei_dev = g[EDGE_TYPES[2]].edge_index


def group_of(name):
    base = name.replace("_missing", "")
    if re.fullmatch(r"V\d+", base): return "V columns (anonymous Vesta features)"
    if re.fullmatch(r"C\d+", base): return "C columns (counts)"
    if re.fullmatch(r"D\d+", base): return "D columns (time gaps)"
    if base.startswith("id_"): return "Identity columns"
    if base == "TransactionAmt": return "Amount"
    if base.endswith("_count"): return "How often the card/address was seen"
    if base.startswith(("card", "addr", "dist")): return "Card and address columns"
    return "Other"


groups = {}
for j, n in enumerate(names):
    groups.setdefault(group_of(n), []).append(j)


class Wrapper(torch.nn.Module):
    """Lets the explainer call our model with plain dictionaries."""
    def __init__(self, model, cat):
        super().__init__()
        self.model, self.cat = model, cat

    def forward(self, x_dict, edge_index_dict):
        d = HeteroData()
        d["transaction"].x = x_dict["transaction"]
        d["transaction"].cat = self.cat
        d["card"].num_nodes = 1
        d["device"].num_nodes = 1
        for et in EDGE_TYPES:                    # a missing edge type is passed as an empty one
            d[et].edge_index = edge_index_dict.get(et, EMPTY)
        return self.model(d)


def neighbourhood(t):
    """The small part of the graph that decides payment t's score."""
    c = int(card_of_tx[t])
    mates_c = (card_of_tx == c).nonzero().flatten()
    dmask = ei_dev[0] == t
    has_dev = bool(dmask.any())
    mates_d = ei_dev[0][ei_dev[1] == int(ei_dev[1][dmask][0])] if has_dev else torch.empty(0, dtype=torch.long)
    nodes = torch.unique(torch.cat([torch.tensor([t]), mates_c, mates_d]))
    remap = torch.full((tx.x.shape[0],), -1, dtype=torch.long)
    remap[nodes] = torch.arange(len(nodes))
    eid = {}
    src = remap[mates_c]
    zero = torch.zeros(len(src), dtype=torch.long)
    eid[EDGE_TYPES[0]], eid[EDGE_TYPES[1]] = torch.stack([src, zero]), torch.stack([zero, src])
    if has_dev:
        src = remap[mates_d]
        zero = torch.zeros(len(src), dtype=torch.long)
        eid[EDGE_TYPES[2]], eid[EDGE_TYPES[3]] = torch.stack([src, zero]), torch.stack([zero, src])
    return dict(t=t, local=int(remap[t]), x=tx.x[nodes], cat=tx.cat[nodes], eid=eid,
                mates_c=mates_c, mates_d=mates_d)


def score(case, x=None, cat=None, eid=None):
    W = Wrapper(model, case["cat"] if cat is None else cat)
    with torch.no_grad():
        out = W({"transaction": case["x"] if x is None else x}, case["eid"] if eid is None else eid)
    return float(out[case["local"]])


def without_links(case):
    out = {}
    for et, e in case["eid"].items():
        tx_row = 0 if et[0] == "transaction" else 1
        out[et] = e[:, e[tx_row] != case["local"]]
    return out


def set_zero(case, cols):
    x = case["x"].clone()
    x[case["local"], cols] = 0
    return x


def keep_only(case, cols):
    x = case["x"].clone()
    row = torch.zeros_like(x[case["local"]])
    row[cols] = x[case["local"]][cols]
    x[case["local"]] = row
    return x


def known_fraud_facts(mates, t):
    mates = mates[mates != t]
    known = tx.train_mask[mates]
    return {"other_payments": int(len(mates)),
            "labelled_payments": int(known.sum()),
            "known_frauds": int((tx.y[mates][known] == 1).sum())}


cases = pd.read_csv("results/cases.csv").head(MAX_CASES)
results, fid_rows = [], []
for n_done, (node, tid) in enumerate(zip(cases["node"], cases["TransactionID"]), 1):
    t0 = time.time()
    case = neighbourhood(int(node))
    local = case["local"]
    base = score(case)
    exact = abs(base - float(scores[node])) < 1e-2
    x_row = case["x"][local]

    # 1. How much of the score comes from the connections?
    graph_part = base - score(case, eid=without_links(case))

    # 2. Groups of features: what happens to the score if a whole group is set to "typical"?
    group_effect = {k: base - score(case, x=set_zero(case, idx)) for k, idx in groups.items()}

    # 3. Category columns: what if one is set to "missing"?
    cat_effect = {}
    for k, cname in enumerate(cat_names):
        c2 = case["cat"].clone()
        c2[local, k] = 0
        cat_effect[cname] = base - score(case, cat=c2)

    # 4. Method A: leave-one-out on single features
    loo = torch.zeros(len(names))
    for j in range(len(names)):
        if float(x_row[j]) != 0.0:
            loo[j] = base - score(case, x=set_zero(case, [j]))

    # 5. Method B: GNNExplainer (feature mask and mask over the card links)
    explainer = Explainer(
        model=Wrapper(model, case["cat"]),
        algorithm=GNNExplainer(epochs=200, lr=0.05, node_feat_size=100.0, edge_size=0.5),
        explanation_type="model", node_mask_type="attributes", edge_mask_type="object",
        model_config=dict(mode="regression", task_level="node", return_type="raw"))
    expl = explainer({"transaction": case["x"]}, case["eid"], index=local)
    gmask = expl.node_mask_dict["transaction"][local]
    emask = expl.edge_mask_dict[EDGE_TYPES[0]]

    # 6. Faithfulness: keep only the top features / remove the top features
    rank = {"gnnexplainer": gmask.topk(TOP_K).indices,
            "leave_one_out": loo.abs().topk(TOP_K).indices}
    fid = {"original": base}
    for m, idx in rank.items():
        fid[f"keep_{m}"] = score(case, x=keep_only(case, idx))
        fid[f"remove_{m}"] = score(case, x=set_zero(case, idx))
    rk, rr = [], []
    for _ in range(5):
        idx = torch.randperm(len(names))[:TOP_K]
        rk.append(score(case, x=keep_only(case, idx)))
        rr.append(score(case, x=set_zero(case, idx)))
    fid["keep_random"], fid["remove_random"] = float(np.mean(rk)), float(np.mean(rr))
    fid_rows.append(fid)

    def top_features(vec, k=TOP_K):
        return [{"feature": names[j], "z_value": round(float(x_row[j]), 3),
                 "importance": round(float(vec[j]), 4)} for j in vec.abs().topk(k).indices.tolist()]

    order = [i for i in emask.argsort(descending=True).tolist() if int(case["mates_c"][i]) != int(node)][:5]
    top_mates = [{"TransactionID": int(tx.tx_id[case["mates_c"][i]]),
                  "mask": round(float(emask[i]), 4),
                  "label": int(tx.y[case["mates_c"][i]]) if bool(tx.train_mask[case["mates_c"][i]]) else None}
                 for i in order]

    results.append({
        "TransactionID": int(tid), "score": round(base, 3), "probability": round(float(torch.sigmoid(torch.tensor(base))), 4),
        "neighbourhood_matches_full_graph": exact,
        "card": known_fraud_facts(case["mates_c"], int(node)),
        "device": known_fraud_facts(case["mates_d"], int(node)) if len(case["mates_d"]) else None,
        "score_from_connections": round(graph_part, 3),
        "feature_group_effect": {k: round(v, 3) for k, v in sorted(group_effect.items(), key=lambda kv: -abs(kv[1]))},
        "category_effect_top5": [{"column": k, "effect": round(v, 3)} for k, v in
                                 sorted(cat_effect.items(), key=lambda kv: -abs(kv[1]))[:5]],
        "top_features_gnnexplainer": top_features(gmask),
        "top_features_leave_one_out": top_features(loo),
        "top_card_links_gnnexplainer": top_mates,
    })
    print(f"[{n_done}/{len(cases)}] TransactionID {tid} score {base:.2f} | from connections {graph_part:.2f} "
          f"| neighbourhood exact: {exact} | {time.time() - t0:.0f}s", flush=True)

with open("results/explanations.json", "w") as f:
    json.dump(results, f, indent=2)

fid = pd.DataFrame(fid_rows)
summary = []
for m in ["gnnexplainer", "leave_one_out", "random"]:
    summary.append({
        "method": m,
        f"score kept with only top {TOP_K} features (% of original)": round(100 * (fid[f"keep_{m}"] / fid["original"]).mean(), 1),
        f"score drop when top {TOP_K} removed (% of original)": round(100 * (1 - fid[f"remove_{m}"] / fid["original"]).mean(), 1),
    })
summary = pd.DataFrame(summary)
summary.to_csv("results/explanation_fidelity.csv", index=False)
print("\n", summary.to_string(index=False))
print("\nconnections explain on average", round(float(np.mean([r["score_from_connections"] / r["score"] for r in results])) * 100, 1),
      "% of the score")
