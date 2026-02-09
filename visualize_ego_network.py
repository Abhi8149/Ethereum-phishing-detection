import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load cleaned data
# -----------------------------
edges = pd.read_csv("transactions_clean.csv")
nodes = pd.read_csv("node_features_labeled.csv")

# -----------------------------
# 2. Build directed graph
# -----------------------------
G = nx.DiGraph()

for _, row in nodes.iterrows():
    G.add_node(row["address"], label=row["label"])

for _, row in edges.iterrows():
    if row["from"] in G and row["to"] in G:
        G.add_edge(row["from"], row["to"])

print("Total nodes:", G.number_of_nodes())
print("Total edges:", G.number_of_edges())

# -----------------------------
# 3. Select phishing node
# -----------------------------
phishing_nodes = nodes[nodes["label"] == 1]["address"].tolist()
target_node = phishing_nodes[0]

print("Selected phishing node:", target_node)

# -----------------------------
# 4. Create 2-hop ego network
# -----------------------------
ego_graph = nx.ego_graph(G, target_node, radius=2)

print("Ego nodes:", ego_graph.number_of_nodes())
print("Ego edges:", ego_graph.number_of_edges())

# -----------------------------
# 5. Layout
# -----------------------------
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(ego_graph, seed=42)

# -----------------------------
# 6. Node coloring
# -----------------------------
node_colors = [
    "red" if ego_graph.nodes[n]["label"] == 1 else "skyblue"
    for n in ego_graph.nodes()
]

nx.draw(
    ego_graph,
    pos,
    node_color=node_colors,
    node_size=250,
    edge_color="gray",
    with_labels=False,
    alpha=0.85
)

# Highlight center phishing node
nx.draw_networkx_nodes(
    ego_graph,
    pos,
    nodelist=[target_node],
    node_color="darkred",
    node_size=500
)

# -----------------------------
# 7. Add labels (SMART labeling)
# -----------------------------

labels = {}

# Always label the phishing node
labels[target_node] = "Phishing Wallet"

# Label top-degree neighbors (max 5)
neighbors = sorted(
    ego_graph.degree,
    key=lambda x: x[1],
    reverse=True
)

for node, degree in neighbors[:6]:
    if node != target_node:
        labels[node] = node[:6] + "..." + node[-4:]

nx.draw_networkx_labels(
    ego_graph,
    pos,
    labels=labels,
    font_size=9,
    font_color="black"
)


plt.title(
    "2-Hop Ego Network of a Phishing Ethereum Address\n"
    "Red = Phishing, Blue = Normal"
)

plt.tight_layout()
plt.savefig("phishing_ego_network_labeled.png", dpi=300)
plt.show()

print("Labeled visualization saved as phishing_ego_network_labeled.png")
