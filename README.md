# Causal Transparency Algebra ($\mathbb{R}_\emptyset$) & $\emptyset$-Net Framework

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24175268.svg)](https://doi.org/10.5281/zenodo.24175268)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Build Status](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

Official implementation of the **Causal Transparency Algebra** ($\mathbb{R}_\emptyset$) and the **$\emptyset$-Net** architecture, authored by **Hakim Guettal**. 

This repository provides a complete software, hardware, and analytical suite for deterministic causal transparency, zero-skip dynamic power reduction in neural processing, and mathematical handling of non-propagating state nullity.

---

## 📌 Executive Summary

Traditional neural networks operate under continuous numeric fields where the zero scalar ($0$) often confuses identity with absence of causal influence. The **Causal Transparency Algebra** ($\mathbb{R}_\emptyset$) introduces an explicit algebraic null element $\emptyset$, extending real field operations to enforce strict causal traceability and dynamic computation dynamic pruning.

### Key Contributions
1. **Mathematical Rigor**: Formal extension of field operators $\oplus_\emptyset$ and $\otimes_\emptyset$ over the domain $\mathbb{R} \cup \{\emptyset\}$.
2. **$\emptyset$-Net Architecture**: Neural layers designed to propagate causal masks dynamically, eliminating redundant matrix calculations.
3. **Hardware Efficiency**: Synthesizable Verilog RTL Multiply-Accumulate (MAC) module featuring immediate clock-gating upon detection of $\emptyset$.
4. **Empirical Benchmarks**: Verified scalability across 1,000,000 synaptic operations with minimal overhead.

---

## 🧮 Mathematical Foundations ($\mathbb{R}_\emptyset$)

Let the extended set be defined as:
$$ \mathbb{R}_\emptyset = \mathbb{R} \cup \{\emptyset\} $$

Where $\emptyset$ represents absolute causal non-existence (distinct from numerical value zero $0.0$).

### Core Operational Rules

1. **Causal Null Addition ($\oplus_\emptyset$)**:
   $$ x \oplus_\emptyset \emptyset = x, \quad \forall x \in \mathbb{R}_\emptyset $$
   $$ \emptyset \oplus_\emptyset \emptyset = \emptyset $$

2. **Causal Null Multiplication ($\otimes_\emptyset$)**:
   $$ x \otimes_\emptyset \emptyset = \emptyset, \quad \forall x \in \mathbb{R}_\emptyset $$

3. **Causal Propagation Invariance**:
   For any tensor mapping function $f: \mathbb{R}_\emptyset^n \to \mathbb{R}_\emptyset^m$, if all inputs $x_i = \emptyset$, then:
   $$ f(\emptyset, \dots, \emptyset) = \emptyset $$

---

## 📁 Repository Structure

```text
rnull-framework/
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   ├── __init__.py
│   ├── rnull_core.py       # Core algebraic operations and tensor classes
│   ├── layers.py           # ∅-Net linear, convolutional, and activation layers
│   └── utils.py            # Transformation and visualization utilities
├── tests/
│   ├── test_algebra.py     # Unit tests for operational rules
│   └── test_layers.py      # Layer-wise forward and backward pass verification
├── hardware/
│   ├── rnull_mac.v         # Synthesizable RTL Verilog MAC module with zero-skip
│   └── testbench_mac.v     # Verilog testbench
├── benchmarks/
│   └── bench_1m.py         # 1 Million Synapses benchmark script
└── web/
    └── index.html          # Interactive web visualization dashboard
```

---

## 🚀 Quick Start & Installation

### Prerequisites
- Python 3.8 or higher
- `numpy`, `pytest` (standard execution dependencies)

### Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/GuettHM/rnull-framework.git
cd rnull-framework
pip install -r requirements.txt
```

### Python Usage Example

```python
from src.rnull_core import RNullTensor
from src.layers import RNullLinear

# Create tensors with explicit causal null state
x = RNullTensor([1.5, 0.0, None], mask=[True, True, False]) # False denotes ∅

# Initialize ∅-Net layer
layer = RNullLinear(in_features=3, out_features=2)

# Forward pass automatically bypasses computation for ∅ states
output = layer(x)
print("Output Tensor:", output)
```

---

## 🔬 Running Tests & Benchmarks

### Executing Unit Tests
To verify mathematical identity and edge case handling across the framework:

```bash
pytest tests/
```

### Executing the 1M-Synapse Benchmark
To run performance profiling on a 1,000,000 synapse layer:

```bash
python benchmarks/bench_1m.py
```

---

## 🛠️ Hardware Integration (Verilog RTL)

The `/hardware/rnull_mac.v` file contains a synthesizable RTL implementation of an $\mathbb{R}_\emptyset$-aware Multiply-Accumulate unit. 

- **Clock-Gating & Zero-Skip**: When input signals match the encoded $\emptyset$ state bit pattern, multiplication logic is bypassed entirely, reducing dynamic power dissipation in hardware accelerators.

To run the simulation using Icarus Verilog:

```bash
iverilog -o mac_sim hardware/rnull_mac.v hardware/testbench_mac.v
vvp mac_sim
```

---

## 📖 How to Cite

If you use the **Causal Transparency Algebra ($\mathbb{R}_\emptyset$)** or **$\emptyset$-Net** framework in your research, please cite this work using the following BibTeX entry:

```bibtex
@software{guettal_2026_rnull,
  author       = {Guettal, Hakim},
  title        = {Causal Transparency Algebra ($\mathbb{R}_\emptyset$) and $\emptyset$-Net Framework},
  month        = nov,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.1},
  doi          = {10.5281/zenodo.24175268},
  url          = {https://doi.org/10.5281/zenodo.24175268}
}
```

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.

**Author**: Hakim Guettal  
**Contact / Repository**: [https://github.com/GuettHM/rnull-framework](https://github.com/GuettHM/rnull-framework)
