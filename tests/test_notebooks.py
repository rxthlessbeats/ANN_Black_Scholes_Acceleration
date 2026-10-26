import ast
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import Sequential

from bs_accel.data import generate_dataset

ROOT = Path(__file__).resolve().parents[1]


class ShortRun(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "generate_dataset":
            node.keywords.append(ast.keyword(arg="n_samples", value=ast.Constant(96)))
        for keyword in node.keywords:
            if keyword.arg in ("epochs", "verbose"):
                keyword.value = ast.Constant(1 if keyword.arg == "epochs" else 0)
        return node


class Trial:
    def __init__(self, number):
        self.number = number
        self.params = {}

    def suggest_int(self, name, low, high):
        self.params[name] = low
        return low

    def suggest_float(self, name, low, high):
        value = low + (high - low) * (self.number + 1) / 3
        self.params[name] = value
        return value


class Study:
    def optimize(self, objective, n_trials):
        results = []
        for number in range(2):
            trial = Trial(number)
            loss = float(objective(trial))
            assert math.isfinite(loss)
            results.append((loss, trial.params))
        self.best_value, self.best_params = min(results, key=lambda result: result[0])


@pytest.mark.parametrize("task,filename", [("pricing", "ann_opt.ipynb"), ("iv", "ann_iv.ipynb")])
def test_notebook_searches_and_checkpoint_restoration(tmp_path, monkeypatch, task, filename):
    # Run real Keras fits; replace only Optuna's trial scheduler and shorten the run.
    optuna = ModuleType("optuna")
    optuna.create_study = lambda **kwargs: Study()
    monkeypatch.setitem(sys.modules, "optuna", optuna)
    monkeypatch.chdir(ROOT / "notebooks")
    notebook = json.loads((ROOT / "notebooks" / filename).read_text())
    scope = {}
    restored_states, test_evaluations = [], []
    original_fit, original_evaluate = Sequential.fit, Sequential.evaluate

    def checked_fit(model, x, y, **kwargs):
        np.testing.assert_array_equal(x, scope["X_train"])
        np.testing.assert_array_equal(y, scope["y_train"])
        val_x, val_y = kwargs["validation_data"]
        np.testing.assert_array_equal(val_x, scope["X_val"])
        np.testing.assert_array_equal(val_y, scope["y_val"])
        for callback in kwargs.get("callbacks", []):
            if isinstance(callback, LearningRateScheduler):
                learning_rate = float(model.optimizer.learning_rate.numpy())
                callback.set_model(model)
                callback.on_epoch_begin(1000)
                model.optimizer.learning_rate.assign(learning_rate)
        return original_fit(model, x, y, **kwargs)

    def checked_evaluate(model, x=None, *args, **kwargs):
        if isinstance(x, np.ndarray) and np.array_equal(x, scope["X_test"]):
            test_evaluations.append(model)
        return original_evaluate(model, x, *args, **kwargs)

    monkeypatch.setattr(Sequential, "fit", checked_fit)
    monkeypatch.setattr(Sequential, "evaluate", checked_evaluate)
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        tree = ast.parse(source)
        compile(tree, filename, "exec")
        if "tensorflow_model_optimization" in source:
            continue
        tree = ast.fix_missing_locations(ShortRun().visit(tree))
        exec(compile(tree, f"{filename}:cell{index}", "exec"), scope)
        if index == 0:
            scope["ROOT"] = tmp_path
            monkeypatch.chdir(tmp_path)
            load_model = scope["load_trained_model"]

            def tracked_load(path):
                model = load_model(path)
                restored_states.append(
                    [v.numpy().copy() for v in (*model.weights, *model.optimizer.variables)]
                )
                return model

            scope["load_trained_model"] = tracked_load

    expected_x, expected_y = generate_dataset(task, n_samples=96)
    np.testing.assert_array_equal(scope["X"], expected_x)
    np.testing.assert_array_equal(scope["y"], expected_y)
    groups = [scope[name][:, 0] for name in ("X_train", "X_val", "X_test")]
    for index, group in enumerate(groups):
        for other in groups[index + 1 :]:
            assert not np.intersect1d(group, other).size
    assert len(test_evaluations) == 1
    assert len(restored_states) == 3  # Two trials, then the selected final run.
    for state in restored_states[1:]:
        for actual, expected in zip(state, restored_states[0], strict=True):
            np.testing.assert_array_equal(actual, expected)
    model = scope["model"]
    assert int(model.optimizer.iterations.numpy()) == 2
    expected_lr = 0.001 * math.exp(scope["learning_study"].best_params["exp_rate"])
    assert float(model.optimizer.learning_rate.numpy()) == pytest.approx(expected_lr)
    assert (tmp_path / "model" / f"notebook-{task}" / "model.keras").exists()


def test_notebook_pruning_compatibility():
    pytest.importorskip("tf_keras")
    pytest.importorskip("tensorflow_model_optimization")
    notebook = json.loads((ROOT / "notebooks" / "ann_opt.ipynb").read_text())
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "tensorflow_model_optimization" in "".join(cell["source"])
    )
    tree = ShortRun().visit(ast.parse(source))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg in ("begin_step", "end_step", "frequency"):
                    keyword.value = ast.Constant(0 if keyword.arg == "begin_step" else 1)
    features, target = generate_dataset("pricing", n_samples=64)
    scope = {
        "exp": math.exp,
        "X_train": features[:48],
        "y_train": target[:48],
        "X_val": features[48:],
        "y_val": target[48:],
    }
    exec(compile(ast.fix_missing_locations(tree), "pruning", "exec"), scope)
    weights = scope["final_pruned_model"].get_weights()
    assert sum(np.count_nonzero(weight == 0) for weight in weights) > 0
