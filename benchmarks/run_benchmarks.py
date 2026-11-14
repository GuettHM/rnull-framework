#!/usr/bin/env python3
"""Reproducible micro-benchmarks for the R_null reference implementation.

Three questions, each answered with measured numbers (NumPy only, seeded):

1. MAC activity  -- what fraction of multiply-accumulate operations is
   skipped when weights and inputs are independently null with rate p?
2. Frozen drift  -- how far do "frozen" weights move under AdamW when their
   gradient is merely zero, compared with a masked update and with null
   gradients?
3. Output scale  -- how does the output scale of a linear layer change with
   the fraction of missing inputs, with and without normalisation?

Usage:
    python benchmarks/run_benchmarks.py            # full run, writes results/
    python benchmarks/run_benchmarks.py --quick    # small sizes, for CI
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from rnull import NullAdamW, NullLinear, RNullArray

RESULTS_DIR = Path(__file__).resolve().parent / "results"


class _Box:
    """Minimal parameter owner for the optimisers."""


# ---------------------------------------------------------------------------
# 1. MAC activity under independent operand sparsity
# ---------------------------------------------------------------------------
def mac_activity(n: int, seeds: int) -> list:
    rows = []
    for p in (0.10, 0.50, 0.70, 0.90):
        active = []
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            w = RNullArray(rng.standard_normal((n, n)), rng.random((n, n)) < p)
            x = RNullArray(rng.standard_normal((n, n)), rng.random((n, n)) < p)
            active.append(int(w.synaptic_mul(x).count_present()))
        total = n * n
        rows.append(
            {
                "operand_null_rate": p,
                "total_macs": total,
                "active_mean": float(np.mean(active)),
                "active_std": float(np.std(active, ddof=1)) if seeds > 1 else 0.0,
                "skipped_pct_measured": 100.0 * (1.0 - np.mean(active) / total),
                "skipped_pct_expected": 100.0 * (1.0 - (1.0 - p) ** 2),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 2. Drift of frozen weights under AdamW
# ---------------------------------------------------------------------------
def frozen_drift(size: int, warmup: int, steps: int, seed: int = 0) -> list:
    rows = []
    for lr, weight_decay in ((1e-3, 1e-2), (1e-3, 2e-2), (1e-2, 1e-1)):
        rng = np.random.default_rng(seed)
        start = rng.standard_normal(size)
        frozen = rng.random(size) < 0.80
        grads = rng.standard_normal((warmup + steps, size))

        def run(mode: str) -> tuple:
            box = _Box()
            box.W = RNullArray(start)
            optimizer = NullAdamW([(box, "W")], lr=lr, weight_decay=weight_decay)
            reference = None
            for t in range(warmup + steps):
                if t == warmup:
                    reference = box.W.v.copy()  # weights are "consolidated" here
                training = t >= warmup
                grad = grads[t].copy()
                if training:
                    grad[frozen] = 0.0
                if training and mode == "null_gradient":
                    optimizer.step([RNullArray(grad, frozen)])
                elif training and mode == "masked_update":
                    snapshot = box.W.v.copy()
                    optimizer.step([RNullArray(grad)])
                    box.W = RNullArray(np.where(frozen, snapshot, box.W.v))
                else:
                    optimizer.step([RNullArray(grad)])
            return box.W.v, reference

        results = {mode: run(mode) for mode in ("zero_gradient", "masked_update", "null_gradient")}
        row = {"lr": lr, "weight_decay": weight_decay, "warmup_steps": warmup, "steps": steps}
        for mode, (final, reference) in results.items():
            row[f"frozen_mse_{mode}"] = float(np.mean((final[frozen] - reference[frozen]) ** 2))
        row["max_abs_diff_masked_vs_null"] = float(
            np.max(np.abs(results["masked_update"][0] - results["null_gradient"][0]))
        )
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# 3. Output scale of a linear layer versus missing-input fraction
# ---------------------------------------------------------------------------
def output_scale(batch: int, seed: int = 0) -> list:
    rows = []
    for missing in (0.0, 0.3, 0.6, 0.9):
        row = {"missing_fraction": missing}
        for mode in ("none", "mean", "variance"):
            rng = np.random.default_rng(seed)
            layer = NullLinear(64, 32, bias=False, normalize=mode, seed=seed)
            x = RNullArray(rng.standard_normal((batch, 64)), rng.random((batch, 64)) < missing)
            out = layer(x)
            row[f"std_{mode}"] = float(out.v[~out.m].std())
        rows.append(row)
    for mode in ("none", "mean", "variance"):
        base = rows[0][f"std_{mode}"]
        for row in rows:
            row[f"relative_std_{mode}"] = row[f"std_{mode}"] / base
    return rows


# ---------------------------------------------------------------------------
def to_markdown(results: dict) -> str:
    lines = ["### MAC activity (independent null rate p on weights and inputs)", ""]
    lines += [
        "| p | MACs | Active (mean ± sd) | Skipped, measured | Skipped, 1 − (1 − p)² |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results["mac_activity"]:
        lines.append(
            f"| {r['operand_null_rate']:.2f} | {r['total_macs']:,} | "
            f"{r['active_mean']:,.0f} ± {r['active_std']:,.0f} | "
            f"{r['skipped_pct_measured']:.2f}% | {r['skipped_pct_expected']:.2f}% |"
        )
    lines += ["", "### Drift of frozen weights under AdamW (MSE against the consolidated values)", ""]
    lines += [
        "| lr | weight decay | Zero gradient | Masked update | Null gradient | max abs diff (masked vs null) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results["frozen_drift"]:
        lines.append(
            f"| {r['lr']:g} | {r['weight_decay']:g} | {r['frozen_mse_zero_gradient']:.3e} | "
            f"{r['frozen_mse_masked_update']:.1e} | {r['frozen_mse_null_gradient']:.1e} | "
            f"{r['max_abs_diff_masked_vs_null']:.1e} |"
        )
    lines += ["", "### Output scale of a 64→32 linear layer (std relative to no missing inputs)", ""]
    lines += [
        "| Missing inputs | normalize=False | normalize=\"mean\" | normalize=\"variance\" |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for r in results["output_scale"]:
        lines.append(
            f"| {r['missing_fraction']:.0%} | {r['relative_std_none']:.2f} | "
            f"{r['relative_std_mean']:.2f} | {r['relative_std_variance']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="small problem sizes (seconds)")
    parser.add_argument("--no-save", action="store_true", help="print only, do not write results/")
    args = parser.parse_args()

    n, seeds, size, steps, batch = (1000, 5, 100_000, 500, 20_000)
    if args.quick:
        n, seeds, size, steps, batch = (200, 2, 5_000, 100, 2_000)

    started = time.perf_counter()
    results = {
        "config": {"n": n, "seeds": seeds, "drift_size": size, "drift_steps": steps, "quick": args.quick},
        "mac_activity": mac_activity(n, seeds),
        "frozen_drift": frozen_drift(size, warmup=50, steps=steps),
        "output_scale": output_scale(batch),
    }
    report = to_markdown(results)
    print(report)
    print(f"(completed in {time.perf_counter() - started:.1f} s)")

    if not args.no_save and not args.quick:
        RESULTS_DIR.mkdir(exist_ok=True)
        (RESULTS_DIR / "core.json").write_text(json.dumps(results, indent=2) + "\n")
        (RESULTS_DIR / "core.md").write_text(report)
        print(f"results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
