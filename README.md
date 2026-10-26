# ANN Black-Scholes Acceleration

Installable package for the notebook ANNs that map Black-Scholes inputs to a call price (`pricing`) or to implied volatility (`iv`).

## Layout

```
src/bs_accel/    data, model, train, evaluate, cli
notebooks/       original research notebooks
tests/           pytest
data/            generated datasets (gitignored)
model/           checkpoints and metrics (gitignored)
```

## Setup

```bash
uv sync
uv run pre-commit install
```

TensorFlow comes from the CPU wheel on PyPI. The GPU extra `tensorflow[and-cuda]` pulls multi-GB NVIDIA libraries and does not fit this machine.

Optuna and the pruning experiment are notebook-only:

```bash
uv sync --extra notebooks
```

## Commands

```bash
uv run black-scholes prepare --task pricing
uv run black-scholes train --task pricing
uv run black-scholes evaluate --task pricing
uv run black-scholes predict --task pricing
uv run black-scholes predict --task pricing --ann

uv run black-scholes prepare --task iv
uv run black-scholes train --task iv
uv run black-scholes evaluate --task iv
uv run black-scholes predict --task iv

uv run black-scholes demo
uv run pytest
```

`prepare` draws a Latin hypercube and prices it with Black-Scholes. No market download. `train` uses the original recorded notebook settings: pricing uses split seed 42, 564 units, 4 hidden layers, batch 1035, then decay `-0.0118279557269848`; IV uses split seed 0, 434 units, 6 hidden layers, batch 1090, then decay `-0.011254927040438942`. Full training uses 1200 epochs, then up to 2000. The updated notebooks rerun selection on validation data and can choose different settings. `demo` runs the pricing path on 256 synthetic rows for one epoch per phase.

`predict --task pricing` uses the exact vectorized formula, needs only prepared data, and does not import TensorFlow. In the reviewed single-thread CPU comparison, the formula was over 600 times faster than ANN inference for 32,768 rows. `--ann` selects the trained pricing network for experiments; `evaluate` always measures the ANN. IV prediction still uses its trained network.

Sample counts must be at least two, and epoch counts must be positive. IV preparation also rejects samples that leave fewer than two rows after filtering. Invalid counts fail before replacing datasets, checkpoints, or metrics.

The first cell of each notebook moves to the repo root and imports `bs_accel`, whether Jupyter starts in `notebooks/` or the repo root.

Both notebooks use the shared seeded generator. They hold out 10% of rows for the final test and take validation rows from the remaining training data. Architecture and learning-rate searches use validation loss. Every learning-rate trial and the final selected run reload the same complete phase-one `.keras` checkpoint, including Adam's state. Pricing uses one learning-rate search over the union of the original two ranges. Notebook checkpoints live under `model/notebook-pricing/` and `model/notebook-iv/`.

The pricing notebook compares warmed ANN and exact inference times. Its optional pruning experiment uses `tf-keras`, which is separate from the Keras 3 models used elsewhere. Old notebook outputs and scores were cleared because the data and selection procedure changed.

The included `option_best.h5` and `iv_best.h5` models load through `bs_accel.evaluate.load_trained_model` for evaluation and prediction. Their legacy optimizer state may not restore; use native `.keras` phase-one checkpoints to restart experiments.

Tests run both notebook searches with tiny datasets, one-epoch fits, and a two-trial stand-in for Optuna. The pruning compatibility test runs when the notebook extras are installed; otherwise it skips.
