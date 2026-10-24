"""Test loss and predictions for a trained notebook MLP."""

from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model


def load_trained_model(path: Path):
    return load_model(path)


def test_loss(model, features, target) -> float:
    loss = model.evaluate(features, target, verbose=0)
    if isinstance(loss, dict):
        loss = loss.get("loss", next(iter(loss.values())))
    elif isinstance(loss, (list, tuple)):
        loss = loss[0]
    return float(loss)


def predict(model, features) -> np.ndarray:
    return np.asarray(model.predict(features, verbose=0)).reshape(-1)
