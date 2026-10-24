"""MLP widths chosen in the notebooks after the Optuna searches."""

from dataclasses import dataclass

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential


@dataclass(frozen=True)
class ModelSpec:
    n_units: int
    n_layers: int
    batch_size: int
    epochs: int
    phase2_epochs: int
    exp_rate: float
    patience: int
    initial_lr: float = 0.001


# pricing: ann_opt.ipynb best width, plus the recorded second-round learning rate.
# iv: ann_iv.ipynb final width and exp_rate.
MODELS: dict[str, ModelSpec] = {
    "pricing": ModelSpec(
        n_units=564,
        n_layers=4,
        batch_size=1035,
        epochs=1200,
        phase2_epochs=2000,
        exp_rate=-0.0118279557269848,
        patience=80,
    ),
    "iv": ModelSpec(
        n_units=434,
        n_layers=6,
        batch_size=1090,
        epochs=1200,
        phase2_epochs=2000,
        exp_rate=-0.011254927040438942,
        patience=100,
    ),
}


def build_mlp(n_features: int, n_units: int, n_layers: int) -> Sequential:
    """Stack from the notebooks: `n_layers` ReLU blocks, then a linear output."""
    model = Sequential()
    # Keras 3 wants Input(shape) in place of the notebooks' input_dim argument.
    model.add(Input(shape=(n_features,)))
    model.add(Dense(n_units, activation="relu", kernel_initializer="glorot_uniform"))
    for _ in range(n_layers - 1):
        model.add(Dense(n_units, activation="relu", kernel_initializer="glorot_uniform"))
    model.add(Dense(1, kernel_initializer="glorot_uniform"))
    model.compile(optimizer="adam", loss="mse")
    return model
