#!/usr/bin/env python3
"""Does null-aware propagation help a model trained on incomplete inputs?

A small MLP is trained on tabular data whose features are removed completely
at random (MCAR) with the same rate at training and test time.  Three ways
of feeding the incomplete input are compared, everything else being equal:

* ``impute``        -- missing entries replaced by the feature mean (zero
                       after standardisation).  This is exactly what
                       ``NullLinear(normalize=False)`` computes.
* ``impute+mask``   -- as above, with the missingness indicators appended as
                       extra input features (a standard strong baseline).
* ``null-mean``     -- ``NullLinear(normalize="mean")``: null inputs are
                       skipped and each sum is rescaled by fan_in / active.
* ``null-variance`` -- ``NullLinear(normalize="variance")``: rescaled by
                       sqrt(fan_in / active).

Rows whose features are all missing produce a null prediction; for scoring
they fall back to the training prior for every method.

Datasets: a synthetic correlated regression problem, and two datasets
bundled with scikit-learn (diabetes: regression; breast cancer:
classification).  scikit-learn is optional; without it only the synthetic
problem runs.

Usage:
    python benchmarks/missing_data.py            # full run, writes results/
    python benchmarks/missing_data.py --quick    # reduced, for CI
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from rnull import NullAdamW, NullMLP, RNullArray

RESULTS_DIR = Path(__file__).resolve().parent / "results"
METHODS = ("impute", "impute+mask", "null-mean", "null-variance")
NORMALIZE = {"impute": False, "impute+mask": False, "null-mean": "mean", "null-variance": "variance"}
RATES = (0.0, 0.2, 0.4, 0.6)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def synthetic(seed: int, n: int = 3000, d: int = 20, latent: int = 5):
    """Correlated features driven by a few latent factors; non-linear target."""
    rng = np.random.default_rng(1000 + seed)
    z = rng.standard_normal((n, latent))
    X = z @ rng.standard_normal((latent, d)) + 0.5 * rng.standard_normal((n, d))
    w = rng.standard_normal(d) / np.sqrt(d)
    y = np.tanh(X @ w) + 0.5 * np.sin(X[:, 0] * X[:, 1] / 2.0) + 0.1 * rng.standard_normal(n)
    return X, y, "regression"


def load_dataset(name: str, seed: int):
    if name == "synthetic":
        return synthetic(seed)
    from sklearn import datasets  # optional dependency

    if name == "diabetes":
        data = datasets.load_diabetes()
        return data.data, data.target.astype(float), "regression"
    if name == "breast_cancer":
        data = datasets.load_breast_cancer()
        return data.data, data.target.astype(float), "classification"
    raise ValueError(f"unknown dataset {name!r}")


def available_datasets() -> list:
    names = ["synthetic"]
    try:
        import sklearn  # noqa: F401

        names += ["diabetes", "breast_cancer"]
    except ImportError:
        pass
    return names


# ---------------------------------------------------------------------------
# One training run
# ---------------------------------------------------------------------------
def prepare(X, y, task, rate, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    X, y = X[order], y[order]
    mask = rng.random(X.shape) < rate
    n_train, n_val = int(0.6 * len(X)), int(0.2 * len(X))
    splits = {
        "train": slice(0, n_train),
        "val": slice(n_train, n_train + n_val),
        "test": slice(n_train + n_val, len(X)),
    }
    # Standardise with statistics of the observed training entries only.
    observed = np.where(mask[splits["train"]], np.nan, X[splits["train"]])
    mean, std = np.nanmean(observed, axis=0), np.nanstd(observed, axis=0)
    std = np.where(std > 0, std, 1.0)
    X = (X - mean) / std
    prior = float(y[splits["train"]].mean())
    scale = 1.0
    if task == "regression":
        scale = float(y[splits["train"]].std())
        y = (y - prior) / scale
    return X, y, mask, splits, prior, scale


def model_input(method, X, mask):
    if method == "impute+mask":
        return RNullArray(np.concatenate([np.where(mask, 0.0, X), mask.astype(float)], axis=1))
    return RNullArray(X, mask)


def score(task, pred: RNullArray, y, fallback):
    """R^2 for regression, accuracy for classification; nulls use ``fallback``."""
    values = np.where(pred.m[:, 0], fallback, pred.v[:, 0])
    if task == "regression":
        return 1.0 - float(np.sum((values - y) ** 2) / np.sum((y - y.mean()) ** 2))
    return float(np.mean((values > 0.0) == (y > 0.5)))


def train_once(method, X, y, mask, splits, task, prior, seed, epochs, hidden=32):
    in_dim = X.shape[1] * (2 if method == "impute+mask" else 1)
    mlp = NullMLP([in_dim, hidden, 1], seed=seed, normalize=NORMALIZE[method])
    optimizer = NullAdamW(mlp.parameters(), lr=1e-2, weight_decay=1e-4)
    data = {k: (model_input(method, X[s], mask[s]), y[s]) for k, s in splits.items()}
    # Fallback for all-missing rows, in the model's output space.
    if task == "regression":
        fallback = 0.0                      # standardised target mean
    else:
        fallback = float(np.log(prior / (1.0 - prior)))  # prior log-odds

    best_val, best_test = -np.inf, np.nan
    x_train, y_train = data["train"]
    for _ in range(epochs):
        out = mlp.forward(x_train)
        z = out.v[:, 0]
        if task == "regression":
            grad = (z - y_train) / len(y_train)
        else:
            grad = (1.0 / (1.0 + np.exp(-z)) - y_train) / len(y_train)
        optimizer.step(mlp.backward(RNullArray(grad[:, None], out.m)))

        val = score(task, mlp.forward(data["val"][0]), data["val"][1], fallback)
        if val > best_val:  # model selection on the validation split
            best_val = val
            best_test = score(task, mlp.forward(data["test"][0]), data["test"][1], fallback)
    return best_test


# ---------------------------------------------------------------------------
def run(datasets, seeds, epochs) -> dict:
    results = {}
    for name in datasets:
        results[name] = {"task": None, "rates": {}}
        for rate in RATES:
            scores = {m: [] for m in METHODS}
            for seed in range(seeds):
                X, y, task = load_dataset(name, seed)
                results[name]["task"] = task
                Xs, ys, mask, splits, prior, _ = prepare(X, y, task, rate, seed)
                for method in METHODS:
                    scores[method].append(
                        train_once(method, Xs, ys, mask, splits, task, prior, seed, epochs)
                    )
            results[name]["rates"][f"{rate:.1f}"] = {
                m: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)), "runs": v}
                for m, v in scores.items()
            }
    return results


def to_markdown(results: dict, seeds: int) -> str:
    lines = []
    for name, entry in results.items():
        metric = "test R²" if entry["task"] == "regression" else "test accuracy"
        lines += [f"### {name} ({metric}, mean ± sd over {seeds} seeds)", ""]
        lines += ["| Missing rate | " + " | ".join(METHODS) + " |", "| ---: |" + " ---: |" * len(METHODS)]
        for rate, by_method in entry["rates"].items():
            cells = [f"{by_method[m]['mean']:.3f} ± {by_method[m]['std']:.3f}" for m in METHODS]
            lines.append(f"| {float(rate):.0%} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="synthetic only, 2 seeds, few epochs")
    parser.add_argument("--no-save", action="store_true", help="print only, do not write results/")
    args = parser.parse_args()

    datasets, seeds, epochs = available_datasets(), 5, 300
    if args.quick:
        datasets, seeds, epochs = ["synthetic"], 2, 40

    started = time.perf_counter()
    results = run(datasets, seeds, epochs)
    report = to_markdown(results, seeds)
    print(report)
    print(f"(completed in {time.perf_counter() - started:.1f} s)")

    if not args.no_save and not args.quick:
        RESULTS_DIR.mkdir(exist_ok=True)
        payload = {"config": {"seeds": seeds, "epochs": epochs, "methods": METHODS, "rates": RATES}, "results": results}
        (RESULTS_DIR / "missing_data.json").write_text(json.dumps(payload, indent=2) + "\n")
        (RESULTS_DIR / "missing_data.md").write_text(report)
        print(f"results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
