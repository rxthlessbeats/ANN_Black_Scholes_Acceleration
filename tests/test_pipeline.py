import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.models import load_model

from bs_accel.cli import main
from bs_accel.data import black_scholes, generate_dataset, save_dataset, split_dataset
from bs_accel.evaluate import load_trained_model
from bs_accel.model import MODELS, build_mlp
from bs_accel.train import train_task


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
    with np.errstate(divide="raise", invalid="raise"):
        features, target = generate_dataset("iv", n_samples=4000, seed=0)
    assert features.shape[1] == 4
    assert len(features) == len(target)
    assert len(target) > 100
    assert features[:, 3].min() > -16.12
    assert features[:, 3].max() < -0.94


def test_iv_sampling_preserves_numpy_random_state():
    state = np.random.get_state()
    try:
        features, target = generate_dataset("iv", n_samples=128, seed=0)
        np.testing.assert_equal(np.random.get_state(), state)
        again_x, again_y = generate_dataset("iv", n_samples=128, seed=0)
        np.testing.assert_array_equal(features, again_x)
        np.testing.assert_array_equal(target, again_y)
    finally:
        np.random.set_state(state)


def test_iv_rejects_too_few_retained_rows():
    with pytest.raises(ValueError, match="fewer than 2"):
        generate_dataset("iv", n_samples=2, seed=8)


@pytest.mark.parametrize("n_layers", (1, 4, 6))
def test_build_mlp_uses_notebook_stack(n_layers):
    assert MODELS["pricing"].n_units == 564
    assert MODELS["pricing"].n_layers == 4
    assert MODELS["iv"].n_units == 434
    assert MODELS["iv"].n_layers == 6
    model = build_mlp(5, n_units=8, n_layers=n_layers)
    dense = [layer for layer in model.layers if layer.__class__.__name__ == "Dense"]
    assert [layer.units for layer in dense] == [8] * n_layers + [1]
    assert dense[0].activation.__name__ == "relu"
    assert dense[-1].activation.__name__ == "linear"
    assert all(layer.kernel_initializer.__class__.__name__ == "GlorotUniform" for layer in dense)


def _run(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(argv)
    assert caught.value.code == 0


@pytest.mark.parametrize("task", ("pricing", "iv"))
@pytest.mark.parametrize("phase", ("1", "both"))
def test_commands(tmp_path, monkeypatch, task, phase):
    monkeypatch.chdir(tmp_path)
    _run(["prepare", "--task", task, "--samples", "64", "--seed", "42"])
    _run(["train", "--task", task, "--epochs", "1", "--phase2-epochs", "1", "--phase", phase])

    checkpoint = tmp_path / "model" / task / "model.keras"
    model = load_trained_model(checkpoint)
    expected_lr = 0.001 * np.exp(MODELS[task].exp_rate if phase == "both" else 0)
    assert float(model.optimizer.learning_rate.numpy()) == pytest.approx(expected_lr)
    assert int(model.optimizer.iterations.numpy()) == (2 if phase == "both" else 1)
    metrics_path = checkpoint.parent / "metrics.json"
    trained_metrics = json.loads(metrics_path.read_text())

    _run(["evaluate", "--task", task])
    metrics = json.loads(metrics_path.read_text())
    assert metrics["test_loss"] == pytest.approx(trained_metrics["test_loss"])
    predict_args = ["predict", "--task", task]
    if task == "pricing":
        predict_args.append("--ann")
    _run(predict_args)
    predictions = np.load(checkpoint.parent / "predictions.npy")
    assert predictions.shape == (metrics["n_test"],)
    assert np.isfinite(predictions).all()


def test_exact_pricing_needs_no_checkpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    features, target = generate_dataset("pricing", n_samples=64)
    save_dataset(tmp_path / "data" / "pricing.npz", features, target)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from bs_accel.cli import main\n"
            "try:\n"
            "    main(['predict', '--task', 'pricing'])\n"
            "except SystemExit as error:\n"
            "    assert error.code == 0\n"
            "assert 'tensorflow' not in sys.modules\n",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    _, _, _, y_test = split_dataset(features, target, "pricing")
    predictions = np.load(tmp_path / "model" / "pricing" / "predictions.npy")
    np.testing.assert_array_equal(predictions, y_test)
    assert not (tmp_path / "model" / "pricing" / "model.keras").exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["prepare", "--samples", "0"],
        ["prepare", "--samples", "1"],
        ["prepare", "--samples", "-1"],
        ["train", "--epochs", "0", "--phase", "1"],
        ["train", "--epochs", "-1"],
        ["train", "--phase2-epochs", "0"],
        ["train", "--phase2-epochs", "-1"],
    ],
)
@pytest.mark.parametrize("task", ("pricing", "iv"))
def test_invalid_counts_preserve_artifacts(tmp_path, monkeypatch, argv, task):
    monkeypatch.chdir(tmp_path)
    features, target = generate_dataset(task, n_samples=16)
    data_path = tmp_path / "data" / f"{task}.npz"
    save_dataset(data_path, features, target)
    model_dir = tmp_path / "model" / task
    model_dir.mkdir(parents=True)
    checkpoint = model_dir / "model.keras"
    checkpoint.write_bytes(b"existing checkpoint")
    metrics = model_dir / "metrics.json"
    metrics.write_text('{"existing": true}\n')
    before = {path: path.read_bytes() for path in (data_path, checkpoint, metrics)}

    with pytest.raises(SystemExit) as caught:
        main([*argv, "--task", task])
    assert caught.value.code == 1
    assert {path: path.read_bytes() for path in before} == before


def test_train_rejects_unknown_phase():
    with pytest.raises(ValueError, match="phase must"):
        train_task("pricing", np.zeros((2, 5)), np.zeros(2), phase="2")


@pytest.mark.parametrize("task,filename", [("pricing", "option_best.h5"), ("iv", "iv_best.h5")])
def test_legacy_models_keep_predictions(task, filename):
    path = Path(__file__).resolve().parents[1] / filename
    reference = load_model(path, compile=False)
    restored = load_trained_model(path)
    features, target = generate_dataset(task, n_samples=32)
    np.testing.assert_array_equal(
        restored(features, training=False).numpy(), reference(features, training=False).numpy()
    )
    assert np.isfinite(restored.evaluate(features, target, verbose=0))


def test_demo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
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
