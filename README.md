# rnull-framework — Causal Transparency Algebra (ℝ<sub>∅</sub>)

[![CI](https://github.com/GuettHM/rnull-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/GuettHM/rnull-framework/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.24215450-blue.svg)](https://doi.org/10.5281/zenodo.24215450)

Arithmetic on real numbers extended with one element, **∅ ("no value was
supplied")**, that is distinct from a measured `0.0` and is the identity of both
addition and multiplication: `x ⊕ ∅ = x` and `x ⊗ ∅ = x`.

The repository contains

* a **NumPy reference implementation** of the algebra, linear layers and
  optimisers (`rnull/`), with no dependency other than NumPy;
* an equivalent **PyTorch implementation** (`rnull/core.py`, optional);
* a **Verilog multiply–accumulate cell** with a 1-bit null tag per operand, a
  bit-accurate Python model of it, and self-checking testbenches (`hardware/`);
* **reproducible benchmarks** with their stored results (`benchmarks/`);
* a **technical note** with definitions, proofs and limits
  ([`paper/rnull_technical_note.pdf`](./paper/rnull_technical_note.pdf)).

**Status: research prototype (alpha).** The semantics are tested; the practical
benefit is modest and is stated plainly below.

## What it is, and what it is not

| It provides | It does not provide |
| :--- | :--- |
| One consistent way to carry "absent" through sums, products and linear layers | Better accuracy than imputation — see [Measured results](#measured-results) |
| An explicit ∅ output when *every* input of a neuron is missing, instead of a bias-only output | Out-of-distribution detection (only entirely missing input is flagged) |
| Exact, bit-for-bit freezing of selected parameter entries | A remedy for catastrophic forgetting (freezing is a masked update) |
| A MAC cell whose result tag separates "nothing accumulated" from "sum is 0" | Synthesis, area, timing or power numbers |

The (value, mask) representation and its rules are closely related to masked
arrays, NaN-skipping reductions and SQL `NULL` aggregates; see
[Relation to existing work](#relation-to-existing-work).

## Installation

```bash
git clone https://github.com/GuettHM/rnull-framework.git
cd rnull-framework
pip install -e ".[dev]"            # NumPy reference + pytest
pip install -e ".[dev,torch]"      # also installs PyTorch
```

Python ≥ 3.9. The package installs as `rnull`.

## Quick start

### NumPy reference

```python
import numpy as np
from rnull import RNullArray, NullLinear, fuse_product, present_count

x    = RNullArray([5.0, 0.0, 7.0], is_null=[False, False, True])   # [5, 0, ∅]
null = RNullArray.null_like(x)

x + null                 # RNullArray([5.0, 0.0, '∅'])   ∅ is the additive identity
x * 2.0                  # RNullArray([10.0, 0.0, 2.0])  ∅ is the multiplicative identity
x.sum()                  # RNullArray(5.0)               reductions skip ∅
x.chi_null()             # array([0., 0., 1.])           1 where ∅

# Product fusion: a missing factor is skipped (and the magnitude changes).
cam, lidar, radar = RNullArray([2.0]), RNullArray([3.0], [True]), RNullArray([4.0])
fuse_product([cam, lidar, radar])        # RNullArray([8.0])
present_count([cam, lidar, radar])       # array([2])  -- keep this next to the value

# Linear layer: skips missing inputs, reports coverage, abstains if all are missing.
layer = NullLinear(4, 2, normalize="variance")
out = layer(RNullArray(np.ones((2, 4)), [[False, True, True, False], [True] * 4]))
out.m                    # [[False, False], [True, True]]   second row: everything missing
layer.last_coverage      # [[0.5, 0.5], [0.0, 0.0]]
```

### PyTorch

```python
import torch
from rnull.core import RNullTensor, RNullLinear, protected_step

layer = RNullLinear(4, 2, normalize="variance")
x = RNullTensor(torch.ones(2, 4), is_null=torch.tensor([[False, True, True, False], [True] * 4]))
out, coverage = layer(x, return_coverage=True)     # out.v, out.m; differentiable in the live inputs

# Freeze individual entries of a parameter while using any stock optimiser:
optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3, weight_decay=1e-2)
frozen = torch.rand_like(layer.weight) < 0.5
out.v.sum().backward()
protected_step(optimizer, [(layer.weight, frozen)])  # frozen entries are unchanged, bit for bit
```

## Semantics

`x`, `y` real; ∅ null. `m` is the null mask of the result.

| Operation | Python | Rule | Result is ∅ when |
| :--- | :--- | :--- | :--- |
| ⊕ addition | `a + b` | `x ⊕ ∅ = x` | both are ∅ |
| ⊖ subtraction | `a - b` | `x ⊖ ∅ = x`, `∅ ⊖ x = −x` | both are ∅ |
| ⊗ fusion product | `a * b` | `x ⊗ ∅ = x`; `0 ⊗ ∅ = 0` | both are ∅ |
| ⊘ division | `a / b` | IEEE-754 on reals (`1/0 = inf`) | either is ∅ |
| ⊙ absorbing product | `w.synaptic_mul(x)` | `w ⊙ ∅ = ∅ ⊙ x = ∅` | either is ∅ |
| ▷ directed coupling | `w.synaptic_skip(x)` | `∅ ▷ x = x`, `w ▷ ∅ = ∅` | `x` is ∅ |
| Σ⊕ / Π⊗ reductions | `a.sum()`, `a.prod()` | skip ∅ | all are ∅ |

Two points are easy to get wrong:

* **A measured zero is not skipped.** `0 ⊗ x = 0`: zero still annihilates a product.
* **▷ is not pruning.** A null weight under ▷ acts as the weight 1 (pass-through).
  A pruned synapse must contribute nothing; that is ⊙, which the linear layers use.

### Structure in brief

Proved in the [technical note](./paper/rnull_technical_note.pdf) and checked by
`tests/test_algebra.py`:

* `(ℝ∅, ⊕)` and `(ℝ∅, ⊗)` are commutative monoids with the same identity ∅ (the
  classical "adjoin an identity" construction). There are no additive inverses:
  `x ⊖ x = 0`, not ∅.
* `⊗` does **not** distribute over `⊕`. For real `a`:
  `(a⊗b ⊕ a⊗c) ⊖ a⊗(b⊕c) = a` if `b` or `c` is ∅, and `0` otherwise. The detector
  `χ∅(x) = ((1⊗0) ⊕ (1⊗x)) ⊖ (1⊗(0⊕x))` is the case `a=1, b=0, c=x`. It equals the
  mask for finite `x` (it returns NaN for `±inf`), so the code reads the mask.
* `(ℝ∅, ⊕, ⊙)` **is** a commutative semiring (zero ∅, unit 1). A linear layer
  `y_i = ⊕_j W_ij ⊙ x_j` is the matrix–vector product over it.
* `▷` is an action of the monoid `(ℝ∅, ⊗)` on `(ℝ∅, ⊕)` by endomorphisms.

## Linear layers

`NullLinear` (NumPy) and `RNullLinear` (PyTorch) compute `y_i = ⊕_j (W_ij ⊙ x_j) ⊕ b_i`.

* A synapse is skipped if its weight or its input is ∅.
* If every synapse of a neuron is skipped the output is ∅ and the bias is not added.
* The **coverage** — the fraction of live synapses per neuron — is returned
  (`layer.last_coverage`, or `return_coverage=True` in PyTorch), because the
  output mask alone does not show that an input was *partly* missing.

Without rescaling (`normalize=False`, the default) the output for a partly
missing input is **exactly** that of an ordinary linear layer on the
zero-imputed input. Two rescalings of the sum over the `k` live synapses out of
`n` are available:

| `normalize` | Factor | Property (inputs missing completely at random) |
| :--- | :--- | :--- |
| `False` | 1 | equals zero imputation; scale shrinks by √(k/n) |
| `"mean"` (or `True`) | n/k | unbiased estimate of the full sum; variance inflated by n/k |
| `"variance"` | √(n/k) | second moment preserved for independent zero-mean inputs |

## Freezing parameter entries

`W ⊕ ∅ = W`: an update that is ∅ on some entries leaves them untouched —
gradient step, momentum and weight decay included. This is a masked update and
gives the same numbers as `W[~mask] += dW[~mask]`.

* NumPy: `NullSGD` / `NullAdamW` skip every entry whose gradient is ∅
  (`layer.consolidated_mask` marks entries to protect).
* PyTorch: `protected_step(optimizer, [(param, mask), ...])` wraps any stock
  optimiser; `continual_learning_step(param, delta, mask)` applies a raw update.

To freeze a *whole* parameter, `requires_grad_(False)` is simpler.

## Hardware

| File | Content |
| :--- | :--- |
| `hardware/rnull_mac.v` | MAC cell `acc ← acc ⊕ (w ⊙ x)`: null tags, operand isolation, clock-enabled accumulator, guard bits, sticky overflow flag, `acc_null` output |
| `hardware/rnull_skip_cell.v` | Pipeline stage using the directed coupling `w ▷ x` |
| `hardware/tb_*.v` | Self-checking testbenches (race-free stimulus, pass/fail summary line) |
| `rnull/hw_model.py` | Bit-accurate Python model of `rnull_mac`, tested against the software algebra |

```bash
iverilog -g2005 -o tb_mac.vvp  hardware/rnull_mac.v       hardware/tb_rnull_mac.v       && vvp tb_mac.vvp
iverilog -g2005 -o tb_skip.vvp hardware/rnull_skip_cell.v hardware/tb_rnull_skip_cell.v && vvp tb_skip.vvp
```

The accumulator is `2·WIDTH + GUARD` bits wide, so at least `2^GUARD`
worst-case products are summed exactly. The cell has **not** been synthesised:
there are no area, timing or power figures, and the "skipped cycles" numbers
below measure switching *activity* only. Zero-skipping MAC cells are standard
in sparse accelerators; the functional addition here is the `acc_null` tag.

## Measured results

Produced by the scripts in `benchmarks/` with the NumPy reference (fixed
seeds); raw output is in [`benchmarks/results/`](./benchmarks/results).

```bash
python benchmarks/run_benchmarks.py     # ~20 s
python benchmarks/missing_data.py       # ~2 min, uses scikit-learn datasets if installed
```

**Skipped MAC operations** — 10⁶ synapses, weights and inputs independently ∅ with rate p, 5 seeds:

| p | Active (mean ± sd) | Skipped, measured | 1 − (1 − p)² |
| ---: | ---: | ---: | ---: |
| 0.10 | 810,047 ± 34 | 19.00% | 19.00% |
| 0.50 | 250,204 ± 176 | 74.98% | 75.00% |
| 0.70 | 89,970 ± 322 | 91.00% | 91.00% |
| 0.90 | 10,011 ± 110 | 99.00% | 99.00% |

**Drift of "frozen" weights under AdamW** — 10⁵ weights, 80% frozen, 500 steps after 50 warm-up steps (MSE against the consolidated values):

| lr | weight decay | Zero gradient only | Masked update | ∅ gradient |
| ---: | ---: | ---: | ---: | ---: |
| 0.001 | 0.01 | 3.0e-05 | 0 | 0 |
| 0.001 | 0.02 | 1.0e-04 | 0 | 0 |
| 0.01 | 0.1 | 1.4e-01 | 0 | 0 |

A zero gradient does not freeze a weight under AdamW, but the drift is small at
usual settings, and a plain masked update removes it just as exactly as ∅ does.

**Output scale of a 64→32 layer** — standard deviation relative to complete inputs:

| Missing inputs | `normalize=False` | `"mean"` | `"variance"` |
| ---: | ---: | ---: | ---: |
| 0% | 1.00 | 1.00 | 1.00 |
| 30% | 0.84 | 1.20 | 1.00 |
| 60% | 0.63 | 1.60 | 1.00 |
| 90% | 0.32 | 3.47 | 1.00 |

**Prediction with incomplete inputs** — one-hidden-layer MLP, features removed
completely at random at the same rate in training and test, mean ± sd over 5
seeds. `impute` is zero imputation of standardised inputs (identical to the
layer with `normalize=False`); `impute+mask` appends the missingness indicators.

| Dataset (metric) | Missing | impute | impute+mask | ∅, `"mean"` | ∅, `"variance"` |
| :--- | ---: | ---: | ---: | ---: | ---: |
| synthetic (R²) | 0% | 0.878 ± 0.038 | 0.875 ± 0.038 | 0.878 ± 0.038 | 0.878 ± 0.038 |
| | 20% | 0.763 ± 0.052 | 0.753 ± 0.044 | 0.753 ± 0.057 | 0.758 ± 0.059 |
| | 40% | 0.694 ± 0.054 | 0.676 ± 0.058 | 0.671 ± 0.061 | 0.684 ± 0.052 |
| | 60% | 0.566 ± 0.073 | 0.556 ± 0.054 | 0.518 ± 0.078 | 0.553 ± 0.083 |
| diabetes (R²) | 0% | 0.389 ± 0.085 | 0.404 ± 0.074 | 0.389 ± 0.085 | 0.389 ± 0.085 |
| | 20% | 0.356 ± 0.105 | 0.378 ± 0.049 | 0.350 ± 0.092 | 0.340 ± 0.118 |
| | 40% | 0.295 ± 0.100 | 0.283 ± 0.069 | 0.227 ± 0.073 | 0.293 ± 0.072 |
| | 60% | 0.224 ± 0.117 | 0.166 ± 0.112 | 0.021 ± 0.108 | 0.173 ± 0.095 |
| breast cancer (accuracy) | 0% | 0.957 ± 0.006 | 0.969 ± 0.020 | 0.957 ± 0.006 | 0.957 ± 0.006 |
| | 20% | 0.936 ± 0.029 | 0.944 ± 0.027 | 0.946 ± 0.011 | 0.937 ± 0.030 |
| | 40% | 0.934 ± 0.019 | 0.944 ± 0.020 | 0.934 ± 0.023 | 0.934 ± 0.019 |
| | 60% | 0.917 ± 0.010 | 0.927 ± 0.032 | 0.911 ± 0.016 | 0.911 ± 0.020 |

**Reading.** The differences are within one standard deviation almost
everywhere: on these problems null-aware rescaling does **not** improve on
plain imputation, and the `"mean"` rescaling is clearly worse at high missing
rates on the small regression set (variance inflation). The experiments are
small, tabular and use completely-at-random missingness; structured or
informative missingness, sequences and larger models are untested.

## When this is useful

* You need **explicit abstention** when a whole input (a sensor, a modality, a
  record) is missing, rather than a prediction driven by the bias.
* You want the **coverage** of each output as a first-class signal for
  downstream logic or monitoring.
* You need to **freeze individual entries** of a tensor exactly under an
  optimiser with weight decay.
* You are prototyping **sparse/event-driven MAC hardware** and want a small,
  tested cell with a result tag and a Python golden model.
* You teach or study missing-value semantics and want a compact, proven
  reference.

If the goal is predictive accuracy with missing features, start with
imputation plus missingness indicators.

## Repository layout

```text
rnull/
  tensor.py      RNullArray: NumPy reference implementation of the algebra
  nn.py          NullLinear, NullMLP, fuse_product, present_count (manual backward)
  optim.py       NullSGD, NullAdamW (a ∅ gradient freezes the entry)
  core.py        RNullTensor, RNullLinear, protected_step (PyTorch, optional)
  hw_model.py    RNullMacModel: bit-accurate model of hardware/rnull_mac.v
tests/           pytest suite (algebraic laws on both backends, layers, optimisers, MAC model)
benchmarks/      run_benchmarks.py, missing_data.py, results/
hardware/        rnull_mac.v, rnull_skip_cell.v and their self-checking testbenches
dashboard/       index.html: interactive browser explorer of the operators (illustrative)
paper/           rnull_technical_note.tex / .pdf
```

## Testing

```bash
pytest                      # PyTorch tests are skipped automatically if torch is absent
pytest -m "not autograd"    # skip the tests that need autograd / torch.optim
```

Continuous integration (`.github/workflows/ci.yml`) runs the suite with and
without PyTorch, simulates both testbenches with Icarus Verilog, and runs the
benchmarks in quick mode.

## Relation to existing work

* **Masked arrays and NaN-skipping reductions** (`numpy.ma`, `nansum`,
  `torch.masked`) and **SQL aggregates** already implement "skip the missing
  value, missing if all are missing". ℝ∅ uses that rule for binary `⊕`/`⊗` as
  well, and names the absorbing alternative `⊙` separately.
* **Partial convolutions** (Liu et al., ECCV 2018) renormalise by the number of
  valid inputs and propagate an updated mask; `normalize="mean"` is the same idea.
* **Missingness indicators** as model inputs (e.g. GRU-D, Che et al., 2018) are
  the standard strong baseline and are included in the benchmark.
* **Zero-skipping accelerators** (e.g. Cnvlutin, SCNN, Eyeriss) skip MAC
  operations for zero operands; `rnull_mac` skips on a tag bit and reports an
  empty sum.

The word "causal" in the project name is historical; the algebra is unrelated
to causal inference.

## Changes from v1.0.1

Version 1.1.0 fixes installation and the test suite, unifies the two
implementations, corrects several bugs, and withdraws claims that the code did
not support. See [`CHANGELOG.md`](./CHANGELOG.md) for the full list.

## Citation

Release v1.1.0 is archived on Zenodo (v1.0.1: `10.5281/zenodo.24175268`):

```bibtex
@software{guettal_2026_rnull,
  author    = {Guettal, Hakim},
  title     = {rnull-framework: Causal Transparency Algebra (R\_null), real arithmetic with a null identity element},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v1.1.0},
  doi       = {10.5281/zenodo.24215450},
  url       = {https://doi.org/10.5281/zenodo.24215450}
}
```

## License

MIT — see [`LICENSE`](./LICENSE).
