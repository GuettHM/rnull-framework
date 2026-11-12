# Causal Transparency Algebra (ℝ_∅) & ∅-Net Framework

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24175268.svg)](https://doi.org/10.5281/zenodo.24175268)

Author: Hakim Guettal  
DOI: [10.5281/zenodo.24175268](https://doi.org/10.5281/zenodo.24175268)  
Repository: [GuettHM/rnull-framework](https://github.com/GuettHM/rnull-framework/releases/tag/v1.0.1)

## Citation

```bibtex
@software{guettal_2026_rnull,
  author       = {Guettal, Hakim},
  title        = {Causal Transparency Algebra (ℝ_∅) and ∅-Net Framework},
  month        = nov,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.1},
  doi          = {10.5281/zenodo.24175268},
  url          = {https://doi.org/10.5281/zenodo.24175268}
}
**License:** MIT  

## Overview
`rnull` is the official reference implementation of **Causal Transparency Algebra ($\mathbb{R}_\emptyset$)** and the **$\emptyset$-Net** neuromorphic architecture. It extends the real numbers with a topologically isolated **Universal Transparency Operator ($\emptyset$)** satisfying:
1. **Universal Bilateral Identity:** $x \oplus \emptyset = x$ and $x \otimes \emptyset = x$.
2. **Quotient Collapse:** $x \oslash \emptyset = \emptyset$, $\emptyset \oslash x = \emptyset$, $\emptyset \oslash 0 = \emptyset$.
3. **Algebraic Transparency Detector ($\chi_\emptyset$):** $\chi_\emptyset(x) = ((1 \otimes 0) \oplus (1 \otimes x)) \ominus (1 \otimes (0 \oplus x)) \in \{0, 1\}$.

## Repository Structure
- `rnull/`: Core Python library (`RNullTensor`, `NullLinear`, `MultimodalProductFusion`, `NullMLP`, `NullSGD`, `NullAdam`).
- `tests/`: Automated unit test suite (`test_axioms_and_nn.py`).
- `benchmarks/`: 1,000,000-synapse Continual Learning and Sparsity benchmark (`run_benchmarks.py`).
- `hardware/`: Synthesizable Verilog RTL (`null_mac.v`) and testbench (`tb_null_mac.v`) for Tri-State / Clock-Gated CMOS implementation.
- `dashboard/`: Interactive standalone HTML5/JS visual simulator (`index.html`).
- `paper/`: LaTeX manuscript source (`causal_transparency_algebra.tex`).

## Quickstart
```bash
# Run the automated unit test suite
python3 -m unittest discover -s tests

# Run the 1,000,000-synapse benchmark
python3 benchmarks/run_benchmarks.py
```
