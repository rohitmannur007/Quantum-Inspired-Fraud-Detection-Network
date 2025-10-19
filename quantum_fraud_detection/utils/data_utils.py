import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

def load_data(path):
    df = pd.read_csv(path)
    # Assume 'Class' is label (0: legit, 1: fraud)
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    return X, y

def preprocess_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

def save_processed_data(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_processed_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)