"""Test loss and predictions for a trained notebook MLP."""

import numpy as np
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model


def load_trained_model(path):
    # Map the legacy loss alias so Keras 3 can load the supplied HDF5 checkpoints.
    return load_model(path, custom_objects={"mse": MeanSquaredError()})


def test_loss(model, features, target) -> float:
    return float(model.evaluate(features, target, verbose=0))


def predict(model, features) -> np.ndarray:
    return np.asarray(model.predict(features, verbose=0)).reshape(-1)
