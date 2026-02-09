import pandas as pd
import pickle
import torch
from torch_geometric.data import Data

edges = pd.read_csv("transactions_clean.csv")
nodes = pd.read_csv("node_features_labeled.csv")

node_index = {addr: i for i, addr in enumerate(nodes["address"])}

edge_index = []

for _, row in edges.iterrows():
    if row["from"] in node_index and row["to"] in node_index:
        edge_index.append([
            node_index[row["from"]],
            node_index[row["to"]]
        ])

edge_index = torch.tensor(edge_index).t().contiguous()

X = torch.tensor(
    nodes[["in_degree", "out_degree", "total_in", "total_out", "balance"]].values,
    dtype=torch.float
)

y = torch.tensor(nodes["label"].values, dtype=torch.long)

data = Data(x=X, edge_index=edge_index, y=y)

with open("pyg_graph.pickle", "wb") as f:
    pickle.dump(data, f)

print("✅ PyTorch Geometric dataset ready")
