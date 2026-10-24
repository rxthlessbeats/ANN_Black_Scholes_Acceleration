"""Synthetic Black-Scholes samples used by the project notebooks."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.stats as si
from scipy.stats.qmc import LatinHypercube
from sklearn.model_selection import train_test_split

# ann_iv.ipynb keeps log time-value rows inside this open interval.
LOG_TIME_VALUE_MIN = -16.12
LOG_TIME_VALUE_MAX = -0.94


@dataclass(frozen=True)
class TaskSpec:
    n_samples: int
    s_over_k: tuple[float, float]
    maturity: tuple[float, float]
    rate: tuple[float, float]
    sigma: tuple[float, float]
    split_seed: int
    lhs_seed: int
    implied_vol: bool
    test_size: float = 0.1


# Ranges, sizes, and split seeds are the ones hardcoded in the notebooks.
# LatinHypercube is seeded with that same seed so `prepare` is repeatable.
TASKS: dict[str, TaskSpec] = {
    "pricing": TaskSpec(
        n_samples=1_000_000,
        s_over_k=(0.4, 1.6),
        maturity=(0.2, 1.1),
        rate=(0.02, 0.1),
        sigma=(0.01, 1.0),
        split_seed=42,
        lhs_seed=42,
        implied_vol=False,
    ),
    "iv": TaskSpec(
        n_samples=1_100_000,
        s_over_k=(0.5, 1.4),
        maturity=(0.05, 1.0),
        rate=(0.0, 0.1),
        sigma=(0.05, 1.0),
        split_seed=0,
        lhs_seed=0,
        implied_vol=True,
    ),
}


def black_scholes(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    return call_price


def _scale(unit_column, bounds):
    low, high = bounds
    return unit_column * (high - low) + low


def generate_dataset(task: str, n_samples: int | None = None, seed: int | None = None):
    spec = TASKS[task]
    n_samples = spec.n_samples if n_samples is None else n_samples
    seed = spec.lhs_seed if seed is None else seed
    if spec.implied_vol:
        # ann_iv.ipynb calls np.random.seed before sampling.
        np.random.seed(seed)

    samples = LatinHypercube(d=4, seed=seed).random(n=n_samples)
    s_over_k = _scale(samples[:, 0], spec.s_over_k)
    maturity = _scale(samples[:, 1], spec.maturity)
    rate = _scale(samples[:, 2], spec.rate)
    sigma = _scale(samples[:, 3], spec.sigma)
    strike = np.ones(n_samples)
    spot = s_over_k * strike
    call_prices = black_scholes(spot, strike, maturity, rate, sigma)

    if not spec.implied_vol:
        features = np.column_stack((spot, strike, maturity, rate, sigma))
        return features, call_prices

    # Time value uses exp(-r T) because the notebooks set K = 1.
    time_value = call_prices - np.maximum(spot - np.exp(-rate * maturity), 0)
    log_time_value = np.log(time_value / strike)
    keep = np.logical_and(
        log_time_value > LOG_TIME_VALUE_MIN,
        log_time_value < LOG_TIME_VALUE_MAX,
    )
    features = np.column_stack((spot[keep], maturity[keep], rate[keep], log_time_value[keep]))
    target = sigma[keep]
    if len(target) == 0:
        raise ValueError("Implied-volatility filter removed every sample")
    return features, target


def split_dataset(features, target, task: str):
    spec = TASKS[task]
    return train_test_split(
        features,
        target,
        test_size=spec.test_size,
        random_state=spec.split_seed,
    )


def save_dataset(path: Path, features, target) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=features, y=target)


def load_dataset(path: Path):
    with np.load(path, allow_pickle=False) as data:
        return data["X"], data["y"]
