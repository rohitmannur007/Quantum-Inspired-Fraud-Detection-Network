import networkx as nx
import numpy as np
import pickle
from torch_geometric.utils import from_networkx

def create_synthetic_graph(num_nodes=1000, num_edges=5000):
    # Synthetic graph: nodes as cardholders, edges as transactions
    G = nx.gnm_random_graph(num_nodes, num_edges)
    # Add random features (e.g., transaction amounts)
    for node in G.nodes():
        G.nodes[node]['feature'] = np.random.rand(5)  # 5-dim features
    for edge in G.edges():
        G.edges[edge]['weight'] = np.random.rand()
    return G

def augment_data_with_graph(X, G):
    # Map data to graph nodes (simplified: assume X rows correspond to nodes)
    num_nodes = len(G.nodes())
    if X.shape[0] > num_nodes:
        X = X[:num_nodes]  # Truncate for demo
    for i, node in enumerate(G.nodes()):
        # ✅ Ensure node features are float32 to avoid MPS float64 issues
        G.nodes[node]['x'] = X[i].astype(np.float32)
    return from_networkx(G)  # Convert to PyG data

def save_graph(G, path):
    with open(path, 'wb') as f:
        pickle.dump(G, f)

def load_graph(path):
    with open(path, 'rb') as f:
        return pickle.load(f)