import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load data
# -----------------------------
edges = pd.read_csv("transactions_clean.csv")
nodes = pd.read_csv("node_features_labeled.csv")

# -----------------------------
# 2. Select a node (phishing preferred)
# -----------------------------
phishing_nodes = nodes[nodes["label"] == 1]["address"].tolist()

if len(phishing_nodes) == 0:
    raise ValueError("No phishing node found")

target = phishing_nodes[0]
print("Selected node:", target)

# -----------------------------
# 3. Build directed graph
# -----------------------------
G = nx.DiGraph()

for _, row in nodes.iterrows():
    G.add_node(row["address"], label=row["label"])

for _, row in edges.iterrows():
    if row["from"] in G and row["to"] in G:
        G.add_edge(row["from"], row["to"])

# -----------------------------
# 4. Extract 1-hop ego network
# -----------------------------
ego = nx.ego_graph(G, target, radius=1)

# -----------------------------
# 5. Visualization 1: Transaction flow graph
# -----------------------------
plt.figure(figsize=(10, 8))

pos = nx.spring_layout(ego, seed=42)

node_colors = [
    "red" if n == target else "skyblue"
    for n in ego.nodes()
]

nx.draw(
    ego,
    pos,
    node_color=node_colors,
    node_size=300,
    edge_color="gray",
    with_labels=False,
    alpha=0.85
)

# Label center node
nx.draw_networkx_labels(
    ego,
    pos,
    labels={target: "Target Wallet"},
    font_size=10,
    font_color="black"
)

plt.title("1-Hop Transaction Flow of Target Wallet")
plt.tight_layout()
plt.savefig("node_transaction_flow.png", dpi=300)
plt.show()


# -----------------------------
# 6. Visualization 2: Behavior summary plot
# -----------------------------
node_row = nodes[nodes["address"] == target].iloc[0]

features = {
    "In Degree": node_row["in_degree"],
    "Out Degree": node_row["out_degree"],
    "Total In": node_row["total_in"],
    "Total Out": node_row["total_out"],
    "Balance": node_row["balance"],
}

plt.figure(figsize=(8, 5))
plt.bar(features.keys(), features.values(), color="steelblue")
plt.xticks(rotation=20)
plt.ylabel("Value")
plt.title("Behavioral Features of Target Wallet")

plt.tight_layout()
plt.savefig("node_behavior_summary.png", dpi=300)
plt.show()

