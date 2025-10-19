from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

def train_gp(X_train, y_train):
    kernel = 1.0 * RBF(1.0)
    gp = GaussianProcessClassifier(kernel=kernel, random_state=42)
    gp.fit(X_train, y_train)
    return gp

def predict_gp(gp, X_test):
    y_pred = gp.predict(X_test)
    y_prob = gp.predict_proba(X_test)[:, 1]  # Uncertainty (prob of fraud)
    return y_pred, y_prob