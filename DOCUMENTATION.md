# Documentation: GNN-Based Phishing Detection in Ethereum

## How the Code Works — A Complete Walkthrough

**Reference Paper:** Ratra et al., *"Graph neural network based phishing account detection in Ethereum"*, The Computer Journal, 2024

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [What Problem Are We Solving?](#2-what-problem-are-we-solving)
3. [How Does the Data Flow?](#3-how-does-the-data-flow)
4. [Step-by-Step Code Walkthrough](#4-step-by-step-code-walkthrough)
   - [Step 1: Environment Setup](#step-1-environment-setup)
   - [Step 2: Data Loading](#step-2-data-loading)
   - [Step 3: Building the Transaction Graph](#step-3-building-the-transaction-graph)
   - [Step 4: GAE_PDNA Model (The Main Model)](#step-4-gae_pdna-model-the-main-model)
   - [Step 5: Training GAE_PDNA](#step-5-training-gae_pdna)
   - [Step 6: MagNet Baseline](#step-6-magnet-baseline)
   - [Step 7: DeepWalk & Node2Vec Baselines](#step-7-deepwalk--node2vec-baselines)
   - [Step 8: Data Balancing (SMOTE)](#step-8-data-balancing-smote)
   - [Step 9: Downstream Classification](#step-9-downstream-classification)
   - [Step 10: Evaluation & Visualisation](#step-10-evaluation--visualisation)
5. [Key Concepts Explained](#5-key-concepts-explained)
6. [Input/Output Summary](#6-inputoutput-summary)
7. [Architecture Diagram](#7-architecture-diagram)
8. [Errors Encountered & Solutions](#8-errors-encountered--solutions)

---

## 1. The Big Picture

Imagine Ethereum as a giant network of bank accounts. Each account (called an **address**) sends and receives money (ETH) to/from other accounts. Some of these accounts are **phishing accounts** — they trick people into sending them money.

**Our goal:** Given this network of transactions, automatically figure out which accounts are phishing accounts and which are legitimate.

**The key insight from the paper:** Instead of looking at each account in isolation, we look at the *pattern of transactions around it* — who it sends money to, who sends money to it, how much, and when. A phishing account behaves differently from a normal account when you look at its neighbourhood in the transaction graph.

### How the paper approaches this:

```
Raw Ethereum Transactions
        ↓
Build a Transaction Graph (nodes = accounts, edges = transactions)
        ↓
Extract features for each node (in-degree, out-degree, balance, etc.)
        ↓
Feed the graph into a Graph Neural Network (GAE_PDNA)
        ↓
GNN produces a 15-dimensional "fingerprint" (embedding) for each account
        ↓
Use these fingerprints to classify: Phishing or Not Phishing
```

---

## 2. What Problem Are We Solving?

### The Challenge

- Ethereum has millions of transactions. Manually checking each account is impossible.
- Phishing accounts often look normal at first glance — they have transactions just like everyone else.
- The data is **highly imbalanced** — there are far more legitimate accounts than phishing accounts (like finding a needle in a haystack).

### The Paper's Solution

The paper proposes a **two-stage approach**:

1. **Stage 1 — Unsupervised Learning:** Train a Graph AutoEncoder (GAE_PDNA) to learn a compact representation (embedding) of each node. This stage does NOT use labels. It learns by trying to reconstruct the graph structure.

2. **Stage 2 — Supervised Classification:** Take those embeddings and feed them into traditional classifiers (AdaBoost, Random Forest, etc.) that ARE trained with labels (phishing / not-phishing).

This two-stage approach is powerful because:
- Stage 1 can learn from the entire graph structure (even unlabelled nodes help)
- Stage 2 only needs a small amount of labelled data
- The embedding captures neighbourhood patterns that raw features miss

---

## 3. How Does the Data Flow?

```
INPUT FILES
├── node_features_labeled.csv   (453 accounts with 5 features + label)
└── transactions_clean.csv      (883 transactions between accounts)

                    ↓ [Cell 8: Build PyG Graph]

PyG Data Object
├── x            : (453, 5)  — normalised node features
├── edge_index   : (2, ~883) — who sends to whom (integer IDs)
├── edge_attr    : (~883, 3) — normalised edge features
└── y            : (453,)    — labels (0=legit, 1=phishing)

                    ↓ [Cell 11: Link Split]

Three Splits (for GAE training)
├── train_data  (70% of edges)
├── val_data    (10% of edges)
└── test_data   (20% of edges)

                    ↓ [Cell 11: GAE_PDNA Training — 200 epochs]

Node Embeddings
└── embeddings_gae : (453, 15) — each node gets a 15-dim vector

                    ↓ [Cell 20: SMOTE Balancing]

Balanced Dataset
├── X_gae_bal : (N, 15)  — balanced embeddings
└── y_gae_bal : (N,)     — balanced labels (50/50 split)

                    ↓ [Cell 22: Classification]

Final Predictions
├── Precision, Recall, F1-Score for each classifier
├── AUC scores
└── Comparison tables (Tables 6, 7, 8 from paper)
```

---

## 4. Step-by-Step Code Walkthrough

### Step 1: Environment Setup

**Cells 3-4** install and import the required libraries.

| Library | What it does |
|---------|-------------|
| `torch` | Deep learning framework (PyTorch) |
| `torch_geometric` | Graph neural network library built on PyTorch |
| `PDNConv` | The specific graph convolution layer from the paper |
| `GraphNorm` | Normalisation layer for graph neural networks |
| `sklearn` | Traditional ML classifiers and evaluation metrics |
| `imblearn` | SMOTE and undersampling for handling class imbalance |
| `node2vec` | Library for DeepWalk / Node2Vec random walk embeddings |

**Why GPU?** Graph neural networks involve many matrix multiplications. A GPU (like Colab's T4) can do these in parallel, making training 10-50x faster.

---

### Step 2: Data Loading

**Cell 6** uploads two CSV files to Colab:

#### `node_features_labeled.csv` — One row per Ethereum address

| Column | Meaning | Example |
|--------|---------|---------|
| `address` | Ethereum wallet address | `0xd0cc2b...` |
| `in_degree` | How many unique accounts sent money TO this address | 11 |
| `out_degree` | How many unique accounts this address sent money TO | 8 |
| `total_in` | Total ETH received (log-scaled) | 6.95 |
| `total_out` | Total ETH sent (log-scaled) | 1.56 |
| `balance` | `total_in - total_out` | 5.39 |
| `label` | 0 = legitimate, 1 = phishing | 1 |

**Intuition:** These 5 features capture the "behaviour profile" of an account:
- Phishing accounts often have **high in-degree** (many victims send money to them)
- Phishing accounts tend to **quickly move money out** (high out-degree, low balance)
- The **balance** is often close to zero because they drain funds immediately

#### `transactions_clean.csv` — One row per transaction

| Column | Meaning |
|--------|---------|
| `from` | Sender address |
| `to` | Receiver address |
| `value` | Amount in ETH (already log-scaled with `log1p`) |
| `timeStamp` | Unix timestamp of the transaction |
| `blockNumber` | Ethereum block number |

---

### Step 3: Building the Transaction Graph

**Cell 8** converts the CSV tables into a **PyTorch Geometric (PyG) graph**.

#### What is a graph in this context?

Think of it like a social network diagram:
- Each **node** (circle) = an Ethereum address
- Each **edge** (arrow) = a transaction from one address to another
- The arrows have a direction (from → to) because money flows one way

#### What the code does:

1. **Address → Integer Mapping:**
   Every address string gets a number (0, 1, 2, ...). This is needed because neural networks work with numbers, not strings.

   ```
   "0xfdd3bfe..." → 0
   "0xd0cc2b2..." → 1
   "0xd8711db..." → 2
   ```

2. **Node Features (5 dimensions):**
   For each node, we take its 5 features and normalise them:
   - **Log-transform** the balance column (to reduce extreme values)
   - **Min-Max scaling** to bring all features into [0, 1] range

   ```
   Raw:   [11, 8, 6.95, 1.56, 5.39]
   Scaled: [0.45, 0.32, 0.78, 0.12, 0.63]
   ```

   **Why normalise?** Neural networks train much better when inputs are in a similar range. Without normalisation, a feature like `total_in = 6.95` would dominate over `in_degree = 11` despite both being important.

3. **Edge Index (who connects to whom):**
   A 2×E matrix where E = number of edges:
   ```
   edge_index = [[0, 1, 3, 4, ...],   ← source nodes
                  [1, 2, 1, 1, ...]]   ← destination nodes
   ```
   This says: node 0 sent to node 1, node 1 sent to node 2, etc.

4. **Edge Attributes (3 dimensions):**
   Each edge has 3 features: `[timeStamp, value, blockNumber]`, also Min-Max normalised.

   **Why include edge attributes?** The paper's key contribution is that PDNConv uses edge attributes to learn *how important each edge is*. A large recent transaction is more suspicious than a small old one.

5. **PyG Data Object:**
   Everything is packed into a single `Data` object:
   ```python
   data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
   ```
   This is the standard format for PyG — all graph operations expect data in this form.

---

### Step 4: GAE_PDNA Model (The Main Model)

**Cell 10** defines the proposed model architecture. This is the core contribution of the paper.

#### What is a Graph AutoEncoder (GAE)?

An **AutoEncoder** is a neural network that learns to compress data and then reconstruct it:

```
Input → [Encoder] → Compressed Representation → [Decoder] → Reconstructed Input
```

A **Graph** AutoEncoder does this for graphs:
- **Encoder:** Takes the graph (nodes + edges) and produces a small embedding for each node
- **Decoder:** Takes two node embeddings and predicts whether an edge exists between them
- **Training goal:** The decoder should correctly predict which edges exist and which don't

**Key insight:** If the encoder is forced to compress each node into just 15 numbers, but those 15 numbers are good enough to reconstruct which edges exist, then those 15 numbers must capture the essential "identity" of each node in the network.

#### The Encoder: PDNConv Blocks

The encoder uses **Pathfinder Discovery Network (PDN) Convolution** layers. Here's how they work:

##### What is Graph Convolution?

In a regular image CNN, a convolution filter slides over pixels and aggregates information from neighbouring pixels. Similarly, a **graph convolution** aggregates information from neighbouring nodes.

For each node `v`:
```
new_feature(v) = AGGREGATE(features of all neighbours of v)
```

##### What makes PDNConv special?

Regular graph convolutions treat all neighbours equally. But in our transaction graph, not all transactions are equally important! A transaction of 100 ETH yesterday is more relevant than a transaction of 0.001 ETH a year ago.

**PDNConv** has a built-in small neural network (MLP) that processes the **edge attributes** (timestamp, value, block number) and outputs a **weight** for each edge. This weight determines how much influence each neighbour's message has.

```
For each edge (u → v):
    weight = MLP(edge_attributes)     ← learned from data
    message = weight × features(u)    ← weighted message
    
For each node v:
    new_features(v) = SUM(all incoming messages)
```

This is the **"Pathfinder Discovery"** mechanism — the network discovers which paths (edges) in the graph are most important for the task.

##### Encoder Architecture (4 blocks + 1 final layer):

```
Input: node features (N × 5)
  │
  ├─ Block 1: PDNConv(5→32)  → PReLU → GraphNorm
  ├─ Block 2: PDNConv(32→32) → PReLU → GraphNorm
  ├─ Block 3: PDNConv(32→32) → PReLU → GraphNorm
  ├─ Block 4: PDNConv(32→32) → PReLU → GraphNorm
  └─ Final:   PDNConv(32→15)
  │
Output: node embeddings (N × 15)
```

**Why 4 blocks?**
Each PDNConv layer aggregates information from 1-hop neighbours. With 4 blocks stacked, information can flow from nodes up to 4 hops away. This means each node's embedding captures the structure of its extended neighbourhood.

```
1 block  → knows about direct neighbours (1-hop)
2 blocks → knows about neighbours' neighbours (2-hop)
3 blocks → 3-hop neighbourhood
4 blocks → 4-hop neighbourhood
```

**What is PReLU?**
An activation function: `PReLU(x) = x if x > 0, else α·x` (where α is learned). It introduces non-linearity — without it, stacking layers would be equivalent to a single linear transformation.

**What is GraphNorm?**
Normalises the node features within each graph to have zero mean and unit variance. This stabilises training and helps the network converge faster. It's similar to BatchNorm in CNNs but designed for graph data.

#### The Decoder: Inner Product

The decoder is deliberately simple:

```
score(i, j) = embedding_i · embedding_j   (dot product)
```

If two nodes have similar embeddings, their dot product is high → the decoder predicts an edge exists between them.

**Why so simple?** The paper argues that the encoder should do the heavy lifting. A complex decoder could "cheat" by memorising edges without the encoder learning good representations.

#### The Full GAE_PDNA Model

```python
class GAE_PDNA:
    encoder = GAE_PDNA_Encoder(...)     # 4 PDN blocks + final PDNConv
    decoder = InnerProductDecoder()     # simple dot product

    def encode(x, edge_index, edge_attr):
        return encoder(x, edge_index, edge_attr)    # → N × 15 embeddings

    def decode(z, edge_index):
        return decoder(z, edge_index)               # → scores for each edge
```

---

### Step 5: Training GAE_PDNA

**Cell 11** trains the model. Here's what happens:

#### 1. Random Link Split (Data Preparation)

Before training, we **hide some edges** from the model:
- **70% of edges → training set** (model sees these)
- **10% of edges → validation set** (for monitoring progress)
- **20% of edges → test set** (for final evaluation)

We also generate **negative edges** — random pairs of nodes that are NOT connected. The model needs to learn to distinguish real edges from fake ones.

```
Real edge:     0 → 1   (label = 1, this transaction happened)
Negative edge: 0 → 99  (label = 0, this transaction never happened)
```

#### 2. Loss Function (BCE Reconstruction Loss)

The model's job is to predict which edges are real and which are fake:

```python
pos_scores = model.decode(z, positive_edges)  # should be HIGH
neg_scores = model.decode(z, negative_edges)  # should be LOW

loss = BCE(pos_scores, 1) + BCE(neg_scores, 0)
```

**Binary Cross-Entropy (BCE)** penalises the model when:
- It gives a low score to a real edge (missed a real transaction)
- It gives a high score to a fake edge (hallucinated a transaction)

#### 3. Training Loop (200 epochs)

Each epoch:
1. **Forward pass:** Compute embeddings → compute edge scores → compute loss
2. **Backward pass:** Calculate gradients (how to adjust each weight to reduce loss)
3. **Update:** Adam optimiser adjusts weights

```
Epoch   1: Train Loss: 1.4523 | Val Loss: 1.3891
Epoch  20: Train Loss: 0.8734 | Val Loss: 0.9012
Epoch  40: Train Loss: 0.6521 | Val Loss: 0.7103
  ...
Epoch 200: Train Loss: 0.3245 | Val Loss: 0.4012
```

The loss decreasing means the model is getting better at reconstructing the graph.

#### 4. Extracting Embeddings

After training, we run the encoder one final time on the FULL graph:

```python
embeddings_gae = model.encode(data.x, data.edge_index, data.edge_attr)
# Shape: (453, 15) — each of the 453 nodes now has a 15-dimensional "fingerprint"
```

These 15 numbers per node capture the structural role of that node in the transaction network. Similar nodes (based on their transaction patterns) will have similar embeddings.

---

### Step 6: MagNet Baseline

**Cell 15** implements a second model for comparison.

#### What is MagNet?

MagNet (Magnetic Network) is designed specifically for **directed graphs**. Regular GCNs treat edges as undirected (if A→B exists, they also consider B→A), which loses important information in a transaction graph (who sent money to whom matters!).

MagNet uses the **magnetic Laplacian** — a mathematical object from physics that encodes direction using complex numbers:
- **Real part:** captures how nodes are connected (like regular GCN)
- **Imaginary part:** captures the direction of edges

```
Regular GCN: A sends to B ≈ B sends to A  (same!)
MagNet:      A sends to B ≠ B sends to A  (different — direction matters!)
```

#### MagNet Parameters (from paper):

| Parameter | Value | Why |
|-----------|-------|-----|
| Hidden channels | 8 | Small network (fewer parameters) |
| q (phase) | 0.15 | Controls how much direction matters. Trainable — the model learns the best value |
| Dropout | 0.8 | Very high dropout to prevent overfitting on small dataset |
| K (filter order) | 2 | Polynomial filter of order 2 (considers 2-hop neighbours) |
| Epochs | 100 | Fewer epochs than GAE_PDNA (simpler model) |

**Why is MagNet a baseline, not the main model?** The paper shows that GAE_PDNA with PDNConv (which uses edge attributes) outperforms MagNet (which only uses edge direction). Knowing *how much* and *when* a transaction happened (edge attributes) is more informative than just knowing the direction.

---

### Step 7: DeepWalk & Node2Vec Baselines

**Cell 18** generates two more sets of embeddings using random-walk methods.

#### How DeepWalk Works (Simple Explanation)

1. **Random Walk:** Start at a node, randomly follow edges to visit neighbouring nodes. This creates a "sentence" of nodes:
   ```
   Walk 1: node_5 → node_12 → node_3 → node_8 → node_12 → ...
   Walk 2: node_5 → node_7 → node_14 → node_3 → ...
   ```

2. **Skip-gram (Word2Vec):** Treat these walks like sentences and nodes like words. Train Word2Vec to predict which nodes appear near each other in walks.

3. **Result:** Nodes that frequently co-occur in random walks get similar embeddings. This captures structural similarity — nodes in the same "community" get similar vectors.

**Parameters:** 200 walks per node, each 30 steps long, embedding dimension = 15.

#### How Node2Vec Differs

Node2Vec adds two parameters that **bias** the random walk:
- **p = 2** (return parameter): Makes it less likely to return to the previous node. Higher p = more exploration.
- **q = 0.5** (in-out parameter): Makes it more likely to visit nodes close to the starting node (BFS-like behaviour). Lower q = more local exploration.

```
DeepWalk (p=1, q=1): Unbiased, pure random walk
Node2Vec (p=2, q=0.5): Biased toward exploring the local neighbourhood more thoroughly
```

#### Why Are These Baselines?

DeepWalk and Node2Vec:
- ❌ Don't use node features (ignore in-degree, balance, etc.)
- ❌ Don't use edge attributes (ignore transaction amounts, timestamps)
- ✅ Only use graph structure (who connects to whom)

GAE_PDNA uses ALL of this information, which is why it performs better.

---

### Step 8: Data Balancing (SMOTE)

**Cell 20** solves a critical problem: **class imbalance**.

#### The Problem

In our dataset:
```
Non-phishing accounts: ~443 (98%)
Phishing accounts:     ~10  (2%)
```

If a classifier simply predicts "not phishing" for every account, it would be 98% accurate! But it would miss ALL phishing accounts — which is the entire point of the system.

#### The Paper's Solution (Two-Step Balancing)

**Step 1: SMOTE (Synthetic Minority Over-sampling Technique)**

SMOTE creates **synthetic phishing samples** by interpolating between existing ones:

```
Real phishing embedding A: [0.3, 0.8, 0.1, ...]
Real phishing embedding B: [0.5, 0.6, 0.3, ...]
                            ↓ interpolate
Synthetic embedding:        [0.4, 0.7, 0.2, ...]  ← new fake phishing sample
```

Target: Increase minority class to 10% of majority class count.

**Step 2: Random Undersampling**

Randomly remove majority class (non-phishing) samples until both classes are equal (1:1 ratio).

```
Before: 443 non-phishing, 10 phishing
After SMOTE: 443 non-phishing, ~44 phishing
After undersample: ~44 non-phishing, ~44 phishing
```

**Why this two-step approach?** 
- SMOTE alone would need to create too many synthetic samples (risky — they might not be realistic)
- Undersampling alone would throw away too much data
- The combination keeps the best of both worlds

---

### Step 9: Downstream Classification

**Cell 22** trains 5 different classifiers on the balanced embeddings.

#### Why Multiple Classifiers?

The paper wants to show that GAE_PDNA embeddings work well regardless of which classifier you use. Each classifier has different strengths:

| Classifier | How it works | Strength |
|------------|-------------|----------|
| **AdaBoost** | Chains many weak decision trees; each one focuses on samples the previous one got wrong | Great for imbalanced data |
| **Random Forest** | Trains 200 independent decision trees and votes | Robust, hard to overfit |
| **Logistic Regression** | Fits a linear decision boundary in the 15-D embedding space | Fast, interpretable |
| **Naive Bayes** | Assumes features are independent, uses Bayes' theorem | Works well with small data |
| **One-Class SVM** | Learns what "normal" (non-phishing) looks like; flags anything unusual as phishing | Doesn't need phishing examples for training |

#### The Classification Pipeline

```
Balanced Embeddings (N × 15)
    ↓
80/20 Train/Test Split (stratified)
    ↓
Train each classifier on training set
    ↓
Predict on test set
    ↓
Compute metrics: Precision, Recall, F1-Score, AUC
```

#### What Do the Metrics Mean?

| Metric | Question it answers | Formula |
|--------|-------------------|---------|
| **Precision** | Of all accounts we flagged as phishing, how many actually were? | TP / (TP + FP) |
| **Recall** | Of all actual phishing accounts, how many did we catch? | TP / (TP + FN) |
| **F1-Score** | Harmonic mean of precision and recall (balanced metric) | 2 × (P × R) / (P + R) |
| **AUC** | How well can the model distinguish phishing from non-phishing overall? | Area under ROC curve |

Where: TP = True Positive, FP = False Positive, FN = False Negative

**For phishing detection, Recall is crucial** — we'd rather have some false alarms (low precision) than miss actual phishing accounts (low recall).

---

### Step 10: Evaluation & Visualisation

#### Comparative Tables (Cell 24)

The paper presents three comparison tables:

**Table 7: Which embedding method is best?** (all using AdaBoost)
```
                    Precision  Recall  F1-Score  AUC
DeepWalk              0.62     0.58     0.60    0.65
Node2Vec              0.65     0.61     0.63    0.68
GAE_PDNA (Proposed)   0.85     0.82     0.83    0.90  ← Winner
```

**Table 8: Which classifier is best?** (all using GAE_PDNA embeddings)
```
                       Precision  Recall  F1-Score  AUC
AdaBoost                0.85      0.82     0.83    0.90  ← Best overall
Random Forest           0.83      0.80     0.81    0.88
Logistic Regression     0.78      0.75     0.76    0.82
Naive Bayes             0.72      0.70     0.71    0.78
One-Class SVM           0.65      0.60     0.62    N/A
```

**Table 6: AUC scores** for link prediction models (GAE_PDNA vs MagNet).

*(Note: Actual numbers will vary based on your dataset and random seed.)*

#### t-SNE Visualisation (Cell 26)

t-SNE compresses the 15-dimensional embeddings into 2D for visualisation:

```
15-D embedding → t-SNE → 2-D point on a scatter plot
```

If the GAE_PDNA embeddings are good, you should see:
- **Blue dots** (non-phishing) clustered together
- **Red dots** (phishing) clustered separately

Good separation in the t-SNE plot confirms that the embeddings have learned to distinguish phishing from non-phishing accounts.

---

## 5. Key Concepts Explained

### Graph Neural Network (GNN)

A neural network designed to work with graph-structured data. Instead of processing a fixed-size grid (like images) or a sequence (like text), GNNs process nodes and edges of arbitrary structure.

**Core operation — Message Passing:**
```
For each node:
    1. Collect messages from all neighbours
    2. Aggregate them (sum, mean, max)
    3. Update the node's own features using the aggregated message
```

After K rounds of message passing, each node's features contain information about its K-hop neighbourhood.

### AutoEncoder

A neural network with an **hourglass shape**:
```
Input (large) → Encoder → Bottleneck (small) → Decoder → Output (large)
```

The bottleneck forces the network to learn a compressed representation. For our graph:
- Input: full graph with 5 features per node
- Bottleneck: 15-dimensional embedding per node
- Output: reconstructed adjacency (which edges exist)

### PDNConv (Pathfinder Discovery Network Convolution)

The paper's key contribution. A graph convolution layer where each edge has a learnable importance weight computed from edge attributes.

```
Standard GCNConv:  message = W × feature(neighbour)
PDNConv:           message = f(edge_attr) × W × feature(neighbour)
                             ↑
                    Learned edge importance
                    (from timestamp, value, block number)
```

`f(edge_attr)` is a small 2-layer MLP:
```
edge_attr (3-dim) → Linear(3→6) → ReLU → Linear(6→1) → Sigmoid → weight ∈ [0, 1]
```

### Link Prediction

The task of predicting whether an edge exists between two nodes. Used to train the autoencoder:
- **Positive sample:** A real edge that exists in the graph
- **Negative sample:** A random pair of nodes with no edge

The model learns embeddings such that connected nodes have high dot-product scores and unconnected nodes have low scores.

### Inner Product Decoder

The simplest possible decoder:
```
score(node_i, node_j) = embedding_i · embedding_j = Σ(embedding_i[k] × embedding_j[k])
```

High score → model predicts an edge exists.
Low score → model predicts no edge.

### SMOTE (Synthetic Minority Over-sampling Technique)

Creates artificial samples of the minority class:
1. Pick a minority sample
2. Find its K nearest neighbours (also minority)
3. Create a new sample on the line segment between them

```
Sample A ●─────────────● Sample B
              ↑
         New synthetic sample
         (random point on the line)
```

---

## 6. Input/Output Summary

### Inputs to the Notebook

| File | Content | Shape |
|------|---------|-------|
| `node_features_labeled.csv` | 453 Ethereum addresses with 5 features + label | 453 rows × 7 cols |
| `transactions_clean.csv` | 883 transactions with value, timestamp, block | 883 rows × 5 cols |

### Outputs from the Notebook

| File | Content | Purpose |
|------|---------|---------|
| `gae_pdna_weights.pth` | Trained GAE_PDNA model weights | Reload model without retraining |
| `magnet_weights.pth` | Trained MagNet model weights | Reload model without retraining |
| `embeddings_gae_pdna.npy` | Node embeddings (453 × 15) | Use for classification or analysis |
| `embeddings_deepwalk.npy` | DeepWalk embeddings (453 × 15) | Baseline comparison |
| `embeddings_node2vec.npy` | Node2Vec embeddings (453 × 15) | Baseline comparison |
| `tsne_gae_pdna.png` | t-SNE scatter plot | Visualise embedding quality |
| `comparison_charts.png` | Bar charts comparing methods | Paper figures |

### Intermediate Data (in memory)

| Variable | Shape | Description |
|----------|-------|-------------|
| `data.x` | (453, 5) | Normalised node features |
| `data.edge_index` | (2, ~883) | Edge connectivity |
| `data.edge_attr` | (~883, 3) | Normalised edge features |
| `data.y` | (453,) | Node labels |
| `embeddings_gae` | (453, 15) | GAE_PDNA output embeddings |
| `X_gae_bal` | (N, 15) | Balanced embeddings for classification |

---

## 7. Architecture Diagram

### GAE_PDNA Encoder (The Main Model)

```
                    Node Features (N × 5)
                           │
                    ┌──────┴──────┐
                    │  PDNConv    │  ← uses edge_attr to compute edge weights
                    │  (5 → 32)  │     via MLP: Linear(3→6)→ReLU→Linear(6→1)→Sigmoid
                    ├─────────────┤
                    │   PReLU     │  ← activation: max(x, αx) where α is learned
                    ├─────────────┤
                    │  GraphNorm  │  ← normalise features across the graph
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │  PDNConv    │
                    │ (32 → 32)  │      × 3 more blocks
                    ├─────────────┤      (same structure)
                    │   PReLU     │
                    ├─────────────┤
                    │  GraphNorm  │
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │  PDNConv    │  ← final projection layer (no activation/norm)
                    │ (32 → 15)  │
                    └──────┬──────┘
                           │
                    Node Embeddings (N × 15)
```

### Full GAE_PDNA Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                    GAE_PDNA                              │
│                                                          │
│   ┌────────────┐         ┌──────────────────────┐       │
│   │  Encoder   │         │  Decoder             │       │
│   │            │         │  (Inner Product)      │       │
│   │ 4×PDNBlock │──Z──────│                      │       │
│   │ + PDNConv  │  (N×15) │  score = z_i · z_j   │       │
│   │            │         │                      │       │
│   └────────────┘         └──────────────────────┘       │
│        ↑                          │                      │
│   (x, edge_index,                 │                      │
│    edge_attr)              edge scores                   │
│                         (real vs fake edges)             │
│                                │                         │
│                    BCE Loss ←──┘                         │
│                (reconstruct graph)                       │
└─────────────────────────────────────────────────────────┘
                           │
                    After training:
                           │
                    ┌──────┴──────┐
                    │  Embeddings │
                    │  (N × 15)  │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ↓            ↓            ↓
         ┌─────────┐ ┌─────────┐ ┌───────────┐
         │AdaBoost │ │ Random  │ │ Logistic  │  ...
         │         │ │ Forest  │ │Regression │
         └────┬────┘ └────┬────┘ └─────┬─────┘
              │           │            │
              ↓           ↓            ↓
         Phishing / Not-Phishing predictions
```

### Complete Data Flow (End-to-End)

```
Ethereum Blockchain
        │
        ↓ (Etherscan API — done in dataset preparation)
        │
  Raw Transactions CSV
        │
        ↓ (cleaning, log-scaling, feature extraction — done in dataset prep)
        │
  ┌─────┴─────┐
  │ node_     │ transactions_
  │ features_ │ clean.csv
  │ labeled   │
  │ .csv      │
  └─────┬─────┘
        │
        ↓ [THIS NOTEBOOK]
        │
  ┌─────────────────────────────────────────────┐
  │                                              │
  │  1. Build PyG Graph                          │
  │     (normalise features, build edge_index)   │
  │                                              │
  │  2. Train GAE_PDNA (200 epochs)              │
  │     (learn 15-D embeddings via link pred.)   │
  │                                              │
  │  3. Generate Baseline Embeddings             │
  │     (DeepWalk, Node2Vec — also 15-D)         │
  │                                              │
  │  4. Balance Classes (SMOTE + undersample)    │
  │                                              │
  │  5. Train Classifiers (AdaBoost, RF, ...)    │
  │                                              │
  │  6. Evaluate & Compare                       │
  │     (Precision, Recall, F1, AUC)             │
  │                                              │
  └─────────────────────────────────────────────┘
        │
        ↓
  Final Output: Which accounts are phishing?
  + Comparison showing GAE_PDNA > baselines
```

---

## Quick Reference: Paper Section → Code Cell Mapping

| Paper Section | Topic | Notebook Cell |
|--------------|-------|---------------|
| Section 4 | Dataset description | Cells 6, 8 |
| Section 5.1.1 | Node features | Cell 8 |
| Section 5.1.2 | Feature normalisation | Cell 8 |
| Section 5.1.4 | GAE_PDNA architecture | Cell 10 |
| Section 5.2 | MagNet baseline | Cell 15 |
| Section 5.3 | DeepWalk, Node2Vec | Cell 18 |
| Section 6.2.1 | SMOTE balancing | Cell 20 |
| Section 6.2.2 | Train/val/test split | Cell 11 |
| Table 6 | AUC comparison | Cell 24 |
| Table 7 | Embedding comparison | Cell 24 |
| Table 8 | Classifier comparison | Cell 24 |
| Figure 4 | t-SNE visualisation | Cell 26 |

---

## 8. Errors Encountered & Solutions

This section documents every runtime error encountered during Colab execution, the root cause, and the exact fix applied.

---

### Error 1: `numpy.dtype size changed` — NumPy Binary Incompatibility

**Full error:**
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility.
Expected 96 from C header, got 88.
```

**Root cause:** Google Colab pre-installs NumPy 2.x, but PyG's C extensions (`torch-scatter`, `torch-sparse`, etc.) are compiled against NumPy 1.x. The internal C struct sizes changed in NumPy 2.0, breaking binary compatibility.

**Fix (Cell 3 — Installation):**
```python
!pip install "numpy<2.0"
```
Added as the **very first** install command, before `torch-geometric`. This ensures all subsequent packages use NumPy 1.x headers.

---

### Error 2: `MagNetConv not available in this PyG version`

**Full warning:**
```
WARNING: MagNetConv not available in this PyG version
PyG    : 2.7.0
```

**Root cause:** `MagNetConv` is not included in standard PyG 2.7.0 from pip. The original code had a `try/except` that set `HAS_MAGNET = False` and fell back to a plain GCN — which loses all edge directionality and doesn't implement MagNet at all.

**Fix (Cell 4 — Imports):** Removed the `MagNetConv` try/except block. Added `scipy.sparse` and `to_scipy_sparse_matrix` imports for the custom implementation.

---

### Error 3: `NameError: name 'HAS_MAGNET' is not defined`

**Full error:**
```
NameError: name 'HAS_MAGNET' is not defined
```

**Root cause:** After fixing Error 2 by removing the `try/except MagNetConv` block from imports, the `HAS_MAGNET` variable no longer existed. But Cell 15 (MagNet model) still began with `if HAS_MAGNET:`.

**Fix (Cell 15 — MagNet model):** Replaced the entire `if HAS_MAGNET: ... else: GCN fallback` block with a **custom MagNet implementation** that:
1. **`_build_magnetic_laplacian()`** — constructs the complex Hermitian Laplacian L_q = I - D^(-1/2) A_q D^(-1/2) where A_q = A * exp(2*pi*i*q)
2. **`MagNetConvLayer`** — applies Chebyshev polynomial filters (order K=2) on complex features
3. **`MagNetLinkPredictor`** — two-layer encoder with trainable q parameter (init 0.15), inner-product decoder

No conditional logic or fallback needed — the custom code always runs correctly.

---

### Error 4: `RuntimeError: Expected all tensors to be on the same device` (CPU vs CUDA)

**Full error:**
```
RuntimeError: Expected all tensors to be on the same device, but got mat1 is on cpu,
different from other tensors on cuda:0 (when checking argument in method wrapper_CUDA_addmm)
```

**Where:** Cell 13 (Extract GAE_PDNA embeddings), when calling `model_gae.encode(data.x, ...)`.

**Root cause:** In Cell 11 (GAE training), `transform(data.cpu())` is called to prepare edge splits. PyG's `.cpu()` mutates the `data` object **in-place**, moving all its tensors to CPU. So by the time Cell 13 runs, `data.x`, `data.edge_index`, `data.edge_attr` are on CPU, but `model_gae` weights remain on CUDA — causing a device mismatch.

**Fix (Cell 13 — Extract embeddings):** Added `data = data.to(device)` at the top of the cell to move the graph data back to GPU before encoding:
```python
data = data.to(device)   # move back after data.cpu() in training cell

model_gae.eval()
with torch.no_grad():
    embeddings_gae = model_gae.encode(
        data.x, data.edge_index, data.edge_attr
    ).cpu().numpy()
```

---

### Summary of All Fixes

| # | Error | Cell | Root Cause | Fix |
|---|-------|------|-----------|-----|
| 1 | `numpy.dtype size changed` | 4 (imports) | NumPy 2.x ABI break with PyG C extensions | `!pip install "numpy<2.0"` as first install |
| 2 | `MagNetConv not available` | 4 (imports) | Not in PyG 2.7.0 pip | Removed try/except; added scipy.sparse imports |
| 3 | `HAS_MAGNET` NameError | 15 (MagNet) | Variable removed in Error 2 fix | Full custom MagNet with magnetic Laplacian |
| 4 | CPU vs CUDA device mismatch | 13 (embeddings) | `data.cpu()` mutates in-place | `data = data.to(device)` before encoding |
