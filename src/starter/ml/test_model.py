import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .model import train_model, compute_model_metrics, inference

def test_train_model():
    X, y = np.random.rand(10, 5), np.random.randint(2, size=10)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_compute_model_metrics():
    y, preds = [1, 1, 0, 1], [0, 1, 0, 1]
    precision, recall, fbeta = compute_model_metrics(y, preds)
    # Assert that the metrics are close to the expected value:
    # precision = 1.0, recall = 0.5, fbeta = 0.8
    assert abs(precision - 1) < 0.01 and abs(recall - 0.666) < 0.01 and abs(fbeta - 0.8) < 0.01


def test_inference():
    X = np.random.rand(10, 5)
    y = np.random.randint(2, size=10)
    model = train_model(X, y)
    pred = inference(model, X)
    assert y.shape == pred.shape
