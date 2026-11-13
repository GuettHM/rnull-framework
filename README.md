# Causal Transparency Algebra (ℝ_∅) & ∅-Net Framework

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24175268.svg)](https://doi.org/10.5281/zenodo.24175268)
[![Release](https://img.shields.io/badge/release-v1.0.1-blue.svg)](https://github.com/GuettHM/rnull-framework/releases/tag/v1.0.1)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Author:** Hakim Guettal  
**Persistent DOI:** [10.5281/zenodo.24175268](https://doi.org/10.5281/zenodo.24175268)  
**Repository:** [https://github.com/GuettHM/rnull-framework](https://github.com/GuettHM/rnull-framework)  
**Version:** `v1.0.1`

---

## 1. Overview

The **Causal Transparency Algebra ($\mathbb{R}_\emptyset$)** and **$\emptyset$-Net Framework** (`rnull-framework`) provide a foundational algebraic and neuromorphic computing architecture designed to solve three structural limitations of conventional real-valued ($\mathbb{R}$) neural networks:

1. **Structural Sparsity vs. Representation Collapse:** In standard IEEE-754 arithmetic, the value $0.0$ conflates two distinct epistemic states: *measured zero magnitude* and *absence of causal signal / abstention*. $\mathbb{R}_\emptyset$ introduces a dedicated formal null element $\emptyset 
otin \mathbb{R}$.
2. **Catastrophic Forgetting in Continual Learning:** By separating causal abstention ($\emptyset$) from zero-valued weights ($0$), $\emptyset$-Net enables drift-free continual learning and native epistemic abstention without overwriting consolidated pathways.
3. **Hardware Dynamic Power Waste:** Standard Multiply-Accumulate (MAC) units expend switching energy even on uninformative or masked inputs. The $\mathbb{R}_\emptyset$ algebra maps directly onto a hardware-level **Zero-Power Computation Bypass** in Verilog RTL.

---

## 2. Mathematical Formulation ($\mathbb{R}_\emptyset$)

Let $\mathbb{R}_\emptyset = \mathbb{R} \cup \{\emptyset\}$ be the extended set of real numbers equipped with a formal null element $\emptyset$ representing **causal transparency (epistemic abstention)**.

### 2.1 Operational Semantics & Axioms

Unlike IEEE-754 `NaN` (which absorbs both addition and multiplication, destroying entire reduction trees) and unlike $0 \in \mathbb{R}$ (which acts as an additive identity inside $\mathbb{R}$), $\mathbb{R}_\emptyset$ is a **non-distributive bi-identity algebraic structure**:

* **Extended Addition ($\oplus$):** $\emptyset$ acts as the **universal causal-transparent identity**, leaving active signals unaltered while yielding $\emptyset$ only when all inputs abstain:
  $$a \oplus b = egin{cases} a + b & 	ext{if } a, b \in \mathbb{R} \ a & 	ext{if } a \in \mathbb{R}, \, b = \emptyset \ b & 	ext{if } a = \emptyset, \, b \in \mathbb{R} \ \emptyset & 	ext{if } a = \emptyset, \, b = \emptyset \end{cases}$$

* **Extended Multiplication ($\otimes$):** $\emptyset$ acts as the **causal-absorbing element**, halting signal propagation across inactive or unactivated synapses:
  $$a \otimes b = egin{cases} a \cdot b & 	ext{if } a, b \in \mathbb{R} \ \emptyset & 	ext{if } a = \emptyset 	ext{ or } b = \emptyset \end{cases}$$

### 2.2 Operational Cayley Tables

| $\oplus$ (Addition) | $b \in \mathbb{R} \setminus \{0\}$ | $0$ | $\emptyset$ |
| :--- | :---: | :---: | :---: |
| **$a \in \mathbb{R} \setminus \{0\}$** | $a + b$ | $a$ | $a$ |
| **$0$** | $b$ | $0$ | $0$ |
| **$\emptyset$** | $b$ | $0$ | $\emptyset$ |

| $\otimes$ (Multiplication) | $b \in \mathbb{R} \setminus \{0\}$ | $0$ | $\emptyset$ |
| :--- | :---: | :---: | :---: |
| **$a \in \mathbb{R} \setminus \{0\}$** | $a \cdot b$ | $0$ | $\emptyset$ |
| **$0$** | $0$ | $0$ | $\emptyset$ |
| **$\emptyset$** | $\emptyset$ | $\emptyset$ | $\emptyset$ |

### 2.3 Controlled Non-Distributivity

In $\mathbb{R}_\emptyset$, distributivity holds over $\mathbb{R}$ ($a, b, c \in \mathbb{R}$), whereas mixed expressions with $\emptyset$ intentionally violate classical distributivity to preserve causal provenance:
* If $a = 5, b = -5, c = \emptyset$:
  $$(a \oplus b) \otimes c = (5 + (-5)) \otimes \emptyset = 0 \otimes \emptyset = \emptyset$$
* If $a = \emptyset, b = 3, c = 0$:
  $$(a \otimes c) \oplus (b \otimes c) = \emptyset \oplus 0 = 0$$
  whereas if $b = \emptyset$, $(\emptyset \otimes 0) \oplus (\emptyset \otimes 0) = \emptyset \oplus \emptyset = \emptyset$ (total abstention is preserved rather than collapsing into a false numerical $0$).

---

## 3. Repository Structure

```text
rnull-framework/
├── rnull/                  # Core Python library implementing ℝ_∅ tensors and ∅-Net layers
├── tests/                  # Automated unit test suite validating algebraic axioms & edge cases
├── benchmarks/             # 1M-synapse (1000x1000) matrix benchmark suite & continual learning tests
├── hardware/               # Synthesizable Verilog RTL MAC module with Zero-Power ∅-Bypass
├── dashboard/              # Interactive web dashboard for visualizing ℝ_∅ signal propagation
├── paper/                  # Formal LaTeX manuscript, Cayley tables, and TikZ circuit diagrams
├── CITATION.cff            # Official citation metadata (synchronized with Zenodo v1.0.1)
├── pyproject.toml          # Python package configuration and dependencies
├── LICENSE                 # MIT License
└── README.md               # Project documentation
```

---

## 4. Installation & Reproducibility

### 4.1 Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/GuettHM/rnull-framework.git
cd rnull-framework
pip install -e .
```

### 4.2 Running Unit Tests

Verify all algebraic axioms (bi-identity behavior, $\otimes$-absorption, $\oplus$-transparency, continual learning isolation, and multimodal fusion):

```bash
pytest tests/
```

### 4.3 Running the 1M-Synapse ($1000 	imes 1000$) Benchmark

Execute the computational benchmark evaluating matrix multiplication, active MAC operations bypassed, and continual learning stability across $1,000,000$ synapses:

```bash
python benchmarks/run_benchmarks.py
```

---

## 5. Hardware Integration (Verilog RTL Neuromorphic MAC)

The `/hardware` directory contains the synthesizable Verilog RTL implementation of the **$\emptyset$-Net Multiply-Accumulate (MAC) unit**.

* **1-Bit Causal Tagging (`is_null` flag):** Each operand carries a dedicated hardware validity/abstention bit alongside its numerical payload.
* **Zero-Power Clock/Operand Gating:** When either input operand asserts `is_null == 1'b1`, the combinatorial multiplier and adder trees are physically gated (`∅-bypass`), preventing transistor switching and reducing dynamic power dissipation to near-zero while forwarding the accumulator state unchanged ($Acc_{t} = Acc_{t-1} \oplus \emptyset = Acc_{t-1}$).

---

## 6. How to Cite

If you use the **Causal Transparency Algebra ($\mathbb{R}_\emptyset$)** or the **$\emptyset$-Net Framework** in your research, please cite the official Zenodo archive:

```bibtex
@software{guettal_2026_rnull,
  author       = {Guettal, Hakim},
  title        = {Causal Transparency Algebra (ℝ_∅) & ∅-Net Framework},
  month        = nov,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.1},
  doi          = {10.5281/zenodo.24175268},
  url          = {https://doi.org/10.5281/zenodo.24175268}
}
```

---

## 7. License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
