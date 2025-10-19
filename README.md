# Quantum-Inspired Fraud Detection Network 🛡️

A next-generation AI system for detecting fraudulent transactions using Graph Neural Networks (GNNs), Gaussian Process (GP) uncertainty modeling, and Quantum-Inspired Graph Optimization. Optimized for Apple Silicon (M1/M2/M3) and scalable for large financial datasets.

## Project Overview

Detecting fraud is challenging due to complex entity relationships and rare-event distribution. This project combines:

- **Graph Neural Networks (GNN)**: Capture structural dependencies among accounts and transactions.
- **Gaussian Process (GP)**: Provides uncertainty-aware predictions for risk-sensitive decision-making.
- **Quantum-Inspired Optimization**: Detects high-risk clusters in the transaction graph.
- **Spark-based Simulation**: Efficiently simulates alerts for large datasets.

## Features

- Hybrid GNN + GP ensemble for accurate fraud prediction.
- Quantum-inspired graph partitioning for enhanced cluster-level detection.
- MPS-compatible float32 conversions for Apple Silicon.
- End-to-end pipeline: Preprocessing → Training → Inference → Alerts.
- Scalable alert generation using Apache Spark.
- Subsampling for memory-efficient training on large graphs.

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/rohitmannur007/quantum_fraud_detection.git
   cd quantum_fraud_detection
   ```

2. Create a virtual environment and install dependencies:

   ```
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. (Optional) Install Apache Spark for alert simulation (on macOS):

   ```
   brew install apache-spark
   brew install openjdk@17
   export JAVA_HOME=/opt/homebrew/opt/openjdk@17
   ```

## Project Structure

The folder structure is organized for clarity: source code in `src/` (with submodules for models and utilities), data in `data/`, outputs in `outputs/`, and root-level config/files. This ensures separation of concerns, easy navigation, and scalability.

```
quantum_fraud_detection/
├── src/                    # Core source code
│   ├── models/             # Model definitions
│   │   ├── gnn_model.py    # Graph Neural Network definition
│   │   └── gp_model.py     # Gaussian Process model functions
│   ├── utils/              # Utility functions
│   │   ├── data_utils.py   # Data loading and preprocessing functions
│   │   ├── graph_utils.py  # Graph construction and utilities
│   │   └── quantum_utils.py # Quantum-inspired graph optimization
│   ├── inference.py        # Run inference and ensemble predictions
│   ├── preprocess.py       # Data preprocessing pipeline
│   ├── simulate_alerts.py  # Simulate fraud alerts using Spark
│   └── train.py            # Train GNN and GP models
├── data/                   # Input datasets
│   └── transactions.csv    # Raw transaction dataset (add more as needed)
├── outputs/                # Generated artifacts
│   ├── gnn_model.pth       # Trained GNN model
│   └── gp_model.pkl        # Trained GP model
├── config.py               # Project configuration (paths, hyperparameters)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Workflow

### 1. Data Loading & Graph Construction

- Load transaction data and features.
- Construct a graph: nodes = accounts, edges = transactions.
- Optional: subsample nodes for memory efficiency on M1/M2/M3.

```python
graph_data = load_graph(GRAPH_PATH)
graph_data.x = graph_data.x.float()  # MPS-compatible
```

### 2. Quantum-Inspired Graph Partitioning

- Partition the graph into clusters to highlight suspicious nodes.

```python
nx_graph = to_networkx(graph_data)
partitions = quantum_inspired_optimization(nx_graph)
print(f"Detected {len(partitions)} high-risk clusters")
```

### 3. Model Training

#### Gaussian Process (GP)

- Provides probabilistic predictions with uncertainty:

```python
gp = train_gp(X_train, y_train)
pickle.dump(gp, open(MODEL_PATH_GP, "wb"))
```

#### Graph Neural Network (GNN)

- Node-level fraud prediction using message passing:

```python
model = GNNModel(in_channels=graph_data.num_features).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
```

- Subsampling reduces memory usage for large graphs.
- Supports Apple Silicon via float32 conversion.

### 4. Inference & Ensemble

- Combine GNN and GP outputs for final fraud probability:

```python
combined_prob = (gnn_pred + gp_prob) / 2
auc = roc_auc_score(y_test, combined_prob)
print(f"ROC-AUC: {auc:.4f}")
```

- Current ROC-AUC: ~0.8785

### 5. Fraud Alert Simulation

- Use Spark for scalable alert generation.
- Returns indices of transactions likely to be fraudulent:

```python
alerts = [i for i, p in enumerate(probs) if p > 0.5]
print(f"Simulated fraud alerts: {alerts[:10]}... (total {len(alerts)})")
```

- Demonstrates cluster-based and uncertainty-aware fraud detection.

## Usage

1. Preprocess data:

   ```
   python -m src.preprocess
   ```

2. Train models:

   ```
   python -m src.train
   ```

3. Run inference:

   ```
   python -m src.inference
   ```

4. Simulate alerts:

   ```
   python -m src.simulate_alerts
   ```

## Notes

- **Apple Silicon (M1/M2/M3)**: Converts all float64 tensors to float32 for MPS backend.
- **Memory Optimization**: Subsampling reduces memory usage for large graphs.
- **Security**: Consider `weights_only=True` when loading GNN model to avoid unsafe pickle execution.
- **Quantum Optimization**: Provides cluster-level insights for fraud detection.
- **Subsampling**: Training with a subset of nodes improves speed and avoids OOM errors on MacBooks.

## Future Improvements

- Integrate multi-modal features: geolocation, device fingerprints, transaction context.
- Real-time streaming alerts for live transaction monitoring.
- Advanced GNN architectures (GraphSAGE, GAT) for better relational learning.
- Explore RAG-style reasoning for rare and complex fraud cases.
- Distributed training on GPU clusters for large-scale transaction graphs.

