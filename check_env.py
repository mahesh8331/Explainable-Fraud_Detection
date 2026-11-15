import sys, torch, torch_geometric, pandas, sklearn, xgboost, lightgbm, shap, networkx

print("Python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("GPU available:", torch.cuda.is_available())
print("torch_geometric:", torch_geometric.__version__)
print("pandas:", pandas.__version__)
print("sklearn:", sklearn.__version__)
print("xgboost:", xgboost.__version__)
print("lightgbm:", lightgbm.__version__)
print("shap:", shap.__version__)
print("networkx:", networkx.__version__)

# Tiny GNN test: 3 nodes, one GraphSAGE layer
from torch_geometric.nn import SAGEConv
x = torch.randn(3, 4)
edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
out = SAGEConv(4, 8)(x, edge_index)
print("GNN test output shape:", tuple(out.shape), "(expected (3, 8))")
