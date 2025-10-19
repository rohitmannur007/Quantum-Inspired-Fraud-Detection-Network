import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import pickle

from models.gnn_model import GNNModel
from models.gp_model import train_gp
from utils.data_utils import load_processed_data
from utils.graph_utils import load_graph
from utils.quantum_utils import quantum_inspired_optimization
from config import (
    PROCESSED_DATA_PATH,
    GRAPH_PATH,
    MODEL_PATH_GNN,
    MODEL_PATH_GP,
    DEVICE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
)

# -------------------------
# Load processed data (subsample for M2)
# -------------------------
print("[INFO] Loading processed data...")
X_train, X_test, y_train, y_test = load_processed_data(PROCESSED_DATA_PATH)

SUBSAMPLE_SIZE = 5000
X_train = X_train[:SUBSAMPLE_SIZE]
y_train = y_train[:SUBSAMPLE_SIZE]
X_test = X_test[:1000]
y_test = y_test[:1000]

# -------------------------
# Load graph and subsample nodes
# -------------------------
print("[INFO] Loading graph data...")
graph_data = load_graph(GRAPH_PATH)
graph_data.y = torch.tensor(y_train[:graph_data.num_nodes], dtype=torch.long)

# Proper subsample: avoid slicing Data object directly
num_nodes_to_use = min(SUBSAMPLE_SIZE, graph_data.num_nodes)
node_indices = torch.arange(num_nodes_to_use)
graph_data.x = graph_data.x[node_indices]
graph_data.y = graph_data.y[node_indices]

if hasattr(graph_data, 'edge_index'):
    mask = (graph_data.edge_index[0] < num_nodes_to_use) & (graph_data.edge_index[1] < num_nodes_to_use)
    graph_data.edge_index = graph_data.edge_index[:, mask]
    if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None:
        graph_data.edge_attr = graph_data.edge_attr[mask]

# -------------------------
# Robust dtype conversion for MPS
# -------------------------
for key in list(graph_data.keys()):
    if isinstance(graph_data[key], torch.Tensor) and graph_data[key].dtype == torch.float64:
        graph_data[key] = graph_data[key].float()

# -------------------------
# Quantum optimization
# -------------------------
print("[INFO] Converting to NetworkX graph...")
nx_graph = to_networkx(graph_data)

print("[INFO] Running quantum-inspired graph optimization...")
partitions = quantum_inspired_optimization(nx_graph)
print(f"[INFO] Optimized partitions: {len(partitions)} fraud clusters")

# -------------------------
# Train Gaussian Process (GP)
# -------------------------
print("[INFO] Training Gaussian Process model...")
gp = train_gp(X_train, y_train)
with open(MODEL_PATH_GP, "wb") as f:
    pickle.dump(gp, f)
print(f"[INFO] GP model saved to {MODEL_PATH_GP}")

# -------------------------
# Train GNN
# -------------------------
print("[INFO] Training Graph Neural Network (GNN)...")
model = GNNModel(in_channels=graph_data.num_features).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loader = DataLoader([graph_data], batch_size=BATCH_SIZE)  # Single graph

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    with tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
        for data in pbar:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            out = model(data)
            loss = torch.nn.functional.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
    print(f"[INFO] Epoch {epoch+1} | Avg Loss: {epoch_loss / len(loader):.4f}")

# -------------------------
# Save trained GNN
# -------------------------
torch.save(model.state_dict(), MODEL_PATH_GNN)
print(f"[INFO] Training complete. Model saved to {MODEL_PATH_GNN}")