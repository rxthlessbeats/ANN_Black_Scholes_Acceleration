"""Two-phase Adam training used after the notebooks picked hyperparameters."""

from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

from bs_accel.data import split_dataset
from bs_accel.model import MODELS, build_mlp


def _identity_lr(_epoch, lr):
    # Keras 3 rejects the numpy/tensor scalars these schedules used to return.
    return float(lr)


def _exponential_lr(exp_rate: float):
    def scheduler(_epoch, lr):
        return float(lr * tf.math.exp(exp_rate))

    return scheduler


def set_learning_rate(model, learning_rate: float) -> None:
    optimizer = model.optimizer
    variable = getattr(optimizer, "learning_rate", None)
    if variable is None:
        variable = optimizer.lr
    if hasattr(variable, "assign"):
        variable.assign(learning_rate)
        return
    tf.keras.backend.set_value(variable, learning_rate)


def train_task(
    task: str,
    features,
    target,
    *,
    epochs: int | None = None,
    phase2_epochs: int | None = None,
    phase: str = "both",
    verbose: int = 0,
):
    spec = MODELS[task]
    x_train, x_test, y_train, y_test = split_dataset(features, target, task)
    model = build_mlp(x_train.shape[1], spec.n_units, spec.n_layers)
    batch_size = min(spec.batch_size, len(x_train))
    phase1_epochs = spec.epochs if epochs is None else epochs
    phase2 = spec.phase2_epochs if phase2_epochs is None else phase2_epochs

    if phase == "1" or phase == "both":
        model.fit(
            x_train,
            y_train,
            epochs=phase1_epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=[LearningRateScheduler(_identity_lr)],
        )
    if phase == "both":
        set_learning_rate(model, spec.initial_lr)
        model.fit(
            x_train,
            y_train,
            epochs=phase2,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=[
                LearningRateScheduler(_exponential_lr(spec.exp_rate)),
                EarlyStopping(
                    monitor="loss",
                    patience=spec.patience,
                    restore_best_weights=True,
                ),
            ],
        )

    loss = model.evaluate(x_test, y_test, verbose=0)
    metrics = {
        "task": task,
        "test_loss": float(loss),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
    }
    return model, metrics


def save_checkpoint(model, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
