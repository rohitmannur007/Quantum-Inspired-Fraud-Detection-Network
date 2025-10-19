from utils.data_utils import load_data, preprocess_data, save_processed_data
from utils.graph_utils import create_synthetic_graph, augment_data_with_graph, save_graph
from config import DATA_PATH, PROCESSED_DATA_PATH, GRAPH_PATH

X, y = load_data(DATA_PATH)
X_train, X_test, y_train, y_test = preprocess_data(X, y)
save_processed_data((X_train, X_test, y_train, y_test), PROCESSED_DATA_PATH)

# Create and augment graph
G = create_synthetic_graph(num_nodes=len(X))
graph_data = augment_data_with_graph(X, G)
save_graph(graph_data, GRAPH_PATH)
print("Preprocessing complete.")