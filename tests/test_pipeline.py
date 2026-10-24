import json
from pathlib import Path

import numpy as np
import pytest

from bs_accel.cli import main
from bs_accel.data import black_scholes, generate_dataset, split_dataset
from bs_accel.model import MODELS, build_mlp


def test_black_scholes_atm_zero_rate():
    price = float(black_scholes(100.0, 100.0, 1.0, 0.0, 0.2))
    assert price == pytest.approx(7.965567, rel=1e-5)


def test_pricing_sample_is_reproducible():
    features, target = generate_dataset("pricing", n_samples=128, seed=42)
    again_x, again_y = generate_dataset("pricing", n_samples=128, seed=42)
    assert features.shape == (128, 5)
    assert np.allclose(features, again_x)
    assert np.allclose(target, again_y)
    assert features[:, 0].min() >= 0.4
    assert features[:, 0].max() <= 1.6
    assert target.min() > 0
    train_x, test_x, _, _ = split_dataset(features, target, "pricing")
    train_again, test_again, _, _ = split_dataset(features, target, "pricing")
    assert len(test_x) < len(train_x)
    assert np.allclose(test_x, test_again)


def test_iv_filter_matches_notebook_bounds():
    features, target = generate_dataset("iv", n_samples=4000, seed=0)
    assert features.shape[1] == 4
    assert len(features) == len(target)
    assert len(target) > 100
    assert features[:, 3].min() > -16.12
    assert features[:, 3].max() < -0.94


def test_build_mlp_uses_notebook_stack():
    assert MODELS["pricing"].n_units == 564
    assert MODELS["pricing"].n_layers == 4
    assert MODELS["iv"].n_units == 434
    assert MODELS["iv"].n_layers == 6
    model = build_mlp(5, n_units=8, n_layers=4)
    dense = [layer for layer in model.layers if layer.__class__.__name__ == "Dense"]
    assert len(dense) == 5
    assert [layer.units for layer in dense] == [8, 8, 8, 8, 1]
    assert dense[0].activation.__name__ == "relu"
    assert dense[-1].activation.__name__ == "linear"


def _run(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(argv)
    assert caught.value.code == 0


def test_commands_and_demo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _run(["prepare", "--task", "pricing", "--samples", "64", "--seed", "42"])
    _run(["train", "--task", "pricing", "--epochs", "1", "--phase", "1"])
    _run(["evaluate", "--task", "pricing"])
    _run(["predict", "--task", "pricing"])
    _run(["demo"])

    metrics = json.loads((tmp_path / "model" / "demo" / "metrics.json").read_text())
    assert metrics["task"] == "pricing"
    assert metrics["test_loss"] >= 0
    assert np.isfinite(metrics["test_loss"])
    assert Path(tmp_path / "data" / "demo.npz").exists()
    assert Path(tmp_path / "model" / "demo" / "model.keras").exists()
    predictions = np.load(tmp_path / "model" / "demo" / "predictions.npy")
    assert predictions.shape == (8,)
    assert np.isfinite(predictions).all()
