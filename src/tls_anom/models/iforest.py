import joblib
from sklearn.ensemble import IsolationForest

def train(X, params):
    model = IsolationForest(**params)
    model.fit(X)
    return model

def save(model, path):
    joblib.dump(model, path)

def load(path):
    return joblib.load(path)
