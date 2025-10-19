from pyspark.sql import SparkSession
from utils.data_utils import load_data
from models.gp_model import predict_gp
import pickle
from config import DATA_PATH, MODEL_PATH_GP

# -------------------------
# Spark session (lower memory)
# -------------------------
spark = SparkSession.builder \
    .appName("FraudAlerts") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .getOrCreate()

# -------------------------
# Load dataset
# -------------------------
df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)

# Subsample to reduce memory usage
SUBSAMPLE_SIZE = 1000
df = df.limit(SUBSAMPLE_SIZE)

# Convert to Pandas safely
X = df.drop('Class').toPandas().astype('float32').values  # Ensure float32 for GP/MPS
y_true = df.select('Class').toPandas().values.ravel()

# -------------------------
# Load GP model
# -------------------------
with open(MODEL_PATH_GP, 'rb') as f:
    gp = pickle.load(f)

# -------------------------
# Predict & generate alerts
# -------------------------
_, probs = predict_gp(gp, X)
alerts = [i for i, p in enumerate(probs) if p > 0.5]

print(f"Simulated fraud alerts: {alerts[:10]}... (total {len(alerts)})")

spark.stop()