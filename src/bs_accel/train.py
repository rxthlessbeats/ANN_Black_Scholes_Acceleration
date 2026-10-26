"""Two-phase Adam training used after the notebooks picked hyperparameters."""

from math import exp
from pathlib import Path

from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

from bs_accel.data import split_dataset
from bs_accel.model import MODELS, build_mlp


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
    phase1_epochs = spec.epochs if epochs is None else epochs
    phase2 = spec.phase2_epochs if phase2_epochs is None else phase2_epochs
    if phase not in ("1", "both"):
        raise ValueError("phase must be '1' or 'both'")
    if phase1_epochs < 1 or phase2 < 1:
        raise ValueError("epochs and phase2_epochs must be positive")

    x_train, x_test, y_train, y_test = split_dataset(features, target, task)
    model = build_mlp(x_train.shape[1], spec.n_units, spec.n_layers)

    model.fit(
        x_train,
        y_train,
        epochs=phase1_epochs,
        batch_size=spec.batch_size,
        verbose=verbose,
    )
    if phase == "both":
        model.fit(
            x_train,
            y_train,
            epochs=phase2,
            batch_size=spec.batch_size,
            verbose=verbose,
            callbacks=[
                LearningRateScheduler(lambda _epoch, lr: lr * exp(spec.exp_rate)),
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
