import torch

DATA_PATH = 'data/creditcard.csv'
PROCESSED_DATA_PATH = 'data/processed_data.pkl'
GRAPH_PATH = 'data/transaction_graph.pkl'
MODEL_PATH_GNN = 'outputs/gnn_model.pth'
MODEL_PATH_GP = 'outputs/gp_model.pkl'

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Quantum params
ANNEALING_STEPS = 1000