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

uv run black-scholes prepare --task iv
uv run black-scholes train --task iv
uv run black-scholes evaluate --task iv
uv run black-scholes predict --task iv

uv run black-scholes demo
uv run pytest
```

`prepare` draws a Latin hypercube and prices it with Black-Scholes. No market download. `pricing` follows `ann_opt.ipynb`: split seed 42, 564 units, 4 hidden layers, batch 1035, then the recorded decay `-0.0118279557269848`. `iv` follows `ann_iv.ipynb`: split seed 0, 434 units, 6 hidden layers, batch 1090, then decay `-0.011254927040438942`. Full `train` uses the notebook epoch counts (1200, then up to 2000). `demo` runs the pricing path on 256 synthetic rows for one epoch per phase.

The first cell of each notebook moves to the repo root and imports `bs_accel`, whether Jupyter starts in `notebooks/` or the repo root.
