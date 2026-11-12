"""
Full Benchmark Suite: 1,000,000 Synapses Continual Learning, Sparsity & 7nm CMOS Energy Model.
Author: Hakim Guettal
"""

import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from rnull import RNullTensor, NullLinear, MultimodalProductFusion, NullMLP, NullSGD


def main():
    print("=" * 80)
    print(" CAUSAL TRANSPARENCY ALGEBRA (R_null) & NULL-NET BENCHMARK SUITE")
    print(" Author: Hakim Guettal")
    print("=" * 80)

    N = 1000
    batch = 64
    layer = NullLinear(N, N, seed=42)
    rng = np.random.default_rng(42)

    x_val = rng.normal(0.0, 1.0, (N, batch))
    x_mask = rng.random((N, batch)) < 0.80  # 80% missing/inactive inputs
    X = RNullTensor(x_val, x_mask)

    t0 = time.perf_counter()
    Y = layer.forward(X)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    m = layer.last_metrics
    print(f"[1] 1000x1000 LAYER FORWARD PASS (80% Sparsity)")
    print(f"    Total MACs     : {m['total_macs']:,}")
    print(f"    Executed MACs  : {m['executed_macs']:,}")
    print(f"    Bypassed MACs  : {m['bypassed_macs']:,} (-{m['sparsity_saving_pct']:.2f}%)")
    print(f"    Execution Time : {dt_ms:.2f} ms")

    # Continual Learning Test on 1,000,000 synapses
    W_ref = layer.W.v.copy()
    mask_taskA = rng.random((N, N)) < 0.50
    layer.consolidated_mask = mask_taskA
    opt = NullSGD([layer], lr=0.05, momentum=0.9, weight_decay=1e-3)

    for _ in range(10):
        grad_out = RNullTensor(rng.normal(0, 0.5, (N, batch)))
        _, grad_W = layer.backward(grad_out)
        opt.step([grad_W])

    mse_forgetting = np.mean((layer.W.v[mask_taskA] - W_ref[mask_taskA]) ** 2)
    print(f"\n[2] CONTINUAL LEARNING ON 1,000,000 SYNAPSES (With Momentum + Weight Decay)")
    print(f"    Protected Task-A Synapses : {np.sum(mask_taskA):,}")
    print(f"    Catastrophic Forgetting   : MSE = {mse_forgetting:.12f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
