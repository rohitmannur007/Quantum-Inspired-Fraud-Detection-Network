# Quantum-Inspired Fraud Detection Network

## Overview
Hybrid model using Gaussian Processes, Graph Embeddings, and simulated quantum annealing for fraud detection in transaction networks.

## Setup
1. Activate venv: `source fraud_detection_env/bin/activate`
2. Install deps: `pip install -r requirements.txt`
3. Place `creditcard.csv` in `data/`

## Run
- Preprocess: `python src/preprocess.py`
- Train: `python src/train.py`
- Inference: `python src/inference.py`
- Simulate alerts: `python src/simulate_alerts.py`

## Metrics
Aim for ROC-AUC > 0.95.