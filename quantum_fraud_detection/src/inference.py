import torch
from models.gnn_model import GNNModel
from models.gp_model import predict_gp
from utils.data_utils import load_processed_data
from utils.graph_utils import load_graph
from sklearn.metrics import roc_auc_score
from config import PROCESSED_DATA_PATH, GRAPH_PATH, MODEL_PATH_GNN, MODEL_PATH_GP, DEVICE
import pickle

# -------------------------
# Load data and graph
# -------------------------
_, X_test, _, y_test = load_processed_data(PROCESSED_DATA_PATH)
graph_data = load_graph(GRAPH_PATH)

# ✅ Robust dtype conversion for MPS (convert any float64 tensor to float32)
for key in list(graph_data.keys()):
    if isinstance(graph_data[key], torch.Tensor) and graph_data[key].dtype == torch.float64:
        graph_data[key] = graph_data[key].float()

graph_data = graph_data.to(DEVICE)

# -------------------------
# Load trained GNN model
# -------------------------
model = GNNModel(in_channels=graph_data.num_features).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH_GNN))
model.eval()

with torch.no_grad():
    gnn_pred = model(graph_data).softmax(dim=1)[:, 1].cpu().numpy()  # Fraud probability

# -------------------------
# Load trained GP model
# -------------------------
with open(MODEL_PATH_GP, 'rb') as f:
    gp = pickle.load(f)

gp_pred, gp_prob = predict_gp(gp, X_test)

# -------------------------
# Ensemble predictions and compute ROC-AUC
# -------------------------
combined_prob = (gnn_pred[:len(X_test)] + gp_prob) / 2  # Align sizes
auc = roc_auc_score(y_test, combined_prob)
print(f"ROC-AUC: {auc:.4f}")