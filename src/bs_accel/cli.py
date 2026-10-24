"""End-to-end commands for the Black-Scholes ANNs."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from bs_accel.data import TASKS, generate_dataset, load_dataset, save_dataset, split_dataset
from bs_accel.evaluate import load_trained_model, predict, test_loss
from bs_accel.train import save_checkpoint, train_task


def repo_root() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "bs_accel").is_dir():
            return candidate
    return cwd


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _require(path: Path, hint: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. {hint}")
    return path


def _paths(root: Path, name: str) -> tuple[Path, Path, Path]:
    return (
        root / "data" / f"{name}.npz",
        root / "model" / name / "model.keras",
        root / "model" / name / "metrics.json",
    )


def cmd_prepare(args) -> int:
    root = repo_root()
    features, target = generate_dataset(args.task, n_samples=args.samples, seed=args.seed)
    data_path, _, _ = _paths(root, args.task)
    save_dataset(data_path, features, target)
    print(f"wrote {data_path} features={features.shape} targets={target.shape}")
    return 0


def cmd_train(args) -> int:
    root = repo_root()
    data_path, checkpoint, metrics_path = _paths(root, args.task)
    _require(data_path, f"Run `black-scholes prepare --task {args.task}` first.")
    features, target = load_dataset(data_path)
    model, metrics = train_task(
        args.task,
        features,
        target,
        epochs=args.epochs,
        phase2_epochs=args.phase2_epochs,
        phase=args.phase,
        verbose=args.verbose,
    )
    save_checkpoint(model, checkpoint)
    metrics["checkpoint"] = str(checkpoint)
    _write_json(metrics_path, metrics)
    print(
        f"task={args.task} test_mse={metrics['test_loss']:.6e} "
        f"n_train={metrics['n_train']} n_test={metrics['n_test']}"
    )
    print(f"wrote {checkpoint}")
    return 0


def cmd_evaluate(args) -> int:
    root = repo_root()
    data_path, checkpoint, metrics_path = _paths(root, args.task)
    _require(data_path, f"Run `black-scholes prepare --task {args.task}` first.")
    _require(checkpoint, f"Run `black-scholes train --task {args.task}` first.")
    features, target = load_dataset(data_path)
    _, x_test, _, y_test = split_dataset(features, target, args.task)
    model = load_trained_model(checkpoint)
    metrics = {
        "task": args.task,
        "test_loss": test_loss(model, x_test, y_test),
        "n_test": int(len(y_test)),
        "checkpoint": str(checkpoint),
    }
    _write_json(metrics_path, metrics)
    print(f"task={args.task} test_mse={metrics['test_loss']:.6e} n_test={metrics['n_test']}")
    return 0


def cmd_predict(args) -> int:
    root = repo_root()
    data_path, checkpoint, _ = _paths(root, args.task)
    _require(data_path, f"Run `black-scholes prepare --task {args.task}` first.")
    _require(checkpoint, f"Run `black-scholes train --task {args.task}` first.")
    features, target = load_dataset(data_path)
    _, x_test, _, _ = split_dataset(features, target, args.task)
    model = load_trained_model(checkpoint)
    predictions = predict(model, x_test)
    out = root / "model" / args.task / "predictions.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, predictions)
    preview = " ".join(f"{value:.6g}" for value in predictions[:5])
    print(f"wrote {out} n={len(predictions)} first={preview}")
    return 0


def cmd_demo(_args) -> int:
    """Pricing path on a few hundred synthetic rows, one epoch per phase."""
    root = repo_root()
    task = "pricing"
    features, target = generate_dataset(task, n_samples=256, seed=TASKS[task].lhs_seed)
    data_path = root / "data" / "demo.npz"
    save_dataset(data_path, features, target)
    model, metrics = train_task(task, features, target, epochs=1, phase2_epochs=1, verbose=0)
    checkpoint = root / "model" / "demo" / "model.keras"
    save_checkpoint(model, checkpoint)
    _, x_test, _, y_test = split_dataset(features, target, task)
    metrics = {
        "task": task,
        "test_loss": test_loss(model, x_test, y_test),
        "n_train": metrics["n_train"],
        "n_test": int(len(y_test)),
        "checkpoint": str(checkpoint),
    }
    _write_json(root / "model" / "demo" / "metrics.json", metrics)
    predictions = predict(model, x_test[:8])
    prediction_path = root / "model" / "demo" / "predictions.npy"
    np.save(prediction_path, predictions)
    print(
        f"demo task={task} samples={len(target)} test_mse={metrics['test_loss']:.6e} "
        f"predictions={len(predictions)}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="black-scholes")
    tasks = sorted(TASKS)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Sample synthetic Black-Scholes data")
    prepare.add_argument("--task", choices=tasks, default="pricing")
    prepare.add_argument("--samples", type=int, default=None)
    prepare.add_argument("--seed", type=int, default=None)
    prepare.set_defaults(func=cmd_prepare)

    train = sub.add_parser("train", help="Train the notebook MLP")
    train.add_argument("--task", choices=tasks, default="pricing")
    train.add_argument("--epochs", type=int, default=None, help="Phase-1 epochs")
    train.add_argument("--phase2-epochs", type=int, default=None)
    train.add_argument("--phase", choices=("1", "both"), default="both")
    train.add_argument("--verbose", type=int, default=0)
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("evaluate", help="Test-set MSE")
    evaluate.add_argument("--task", choices=tasks, default="pricing")
    evaluate.set_defaults(func=cmd_evaluate)

    predict_parser = sub.add_parser("predict", help="Predict the held-out split")
    predict_parser.add_argument("--task", choices=tasks, default="pricing")
    predict_parser.set_defaults(func=cmd_predict)

    demo = sub.add_parser("demo", help="Smoke test on synthetic pricing data")
    demo.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        code = args.func(args)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()
