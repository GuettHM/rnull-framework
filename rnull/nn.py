"""
Neural Network Layers over Causal Transparency Algebra (Null-Net).
Author: Hakim Guettal
"""

import numpy as np
from .tensor import RNullTensor, causal_coupling


class MultimodalProductFusion:
    """
    Branchless Multiplicative Sensor Fusion Layer:
    Y = X_1 (*) X_2 (*) ... (*) X_K
    Offline sensors emitting null drop out transparently without zeroing valid channels.
    """
    def forward(self, modalities: list) -> RNullTensor:
        if not modalities:
            raise ValueError("Modalities list cannot be empty.")
        out = modalities[0].copy()
        for mod in modalities[1:]:
            out = out.otimes(mod)
        return out


class NullLinear:
    """
    Null-Net Linear Layer supporting:
    1. Input zero-power bypass when X_j = null.
    2. Passive wire skip-connection when W_ij = null (in residual/multiplicative mode).
    3. Native Epistemic Abstention (output collapses to null if all inputs are null).
    """
    def __init__(self, in_features: int, out_features: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        std = np.sqrt(2.0 / (in_features + out_features))
        self.W = RNullTensor(rng.normal(0.0, std, (out_features, in_features)))
        self.b = RNullTensor(np.zeros((out_features, 1), dtype=np.float64))
        self.consolidated_mask = np.zeros((out_features, in_features), dtype=bool)
        self.last_input = None
        self.last_metrics = {}

    def forward(self, X: RNullTensor) -> RNullTensor:
        self.last_input = X
        total_macs = self.W.v.size * X.v.shape[1]

        active_W = (~self.W.m).astype(np.float64)
        active_X = (~X.m).astype(np.float64)
        active_branches = active_W @ active_X

        executed_macs = int(np.sum(active_branches))
        bypassed_macs = total_macs - executed_macs

        W_eff = np.where(self.W.m, 0.0, self.W.v)
        X_eff = np.where(X.m, 0.0, X.v)
        Z_val = W_eff @ X_eff
        Z_mask = (active_branches == 0)

        Z = RNullTensor(Z_val, Z_mask)
        b_gated = RNullTensor(np.broadcast_to(self.b.v, Z.shape), Z_mask | self.b.m)
        out = Z.oplus(b_gated)

        self.last_metrics = {
            "total_macs": total_macs,
            "executed_macs": executed_macs,
            "bypassed_macs": bypassed_macs,
            "sparsity_saving_pct": (bypassed_macs / total_macs) * 100.0 if total_macs > 0 else 0.0,
        }
        return out

    def backward(self, grad_out: RNullTensor):
        X = self.last_input
        active_g = (~grad_out.m).astype(np.float64)
        active_X = (~X.m).astype(np.float64)
        active_grad_branches = active_g @ active_X.T

        g_eff = np.where(grad_out.m, 0.0, grad_out.v)
        X_eff = np.where(X.m, 0.0, X.v)

        grad_W_val = g_eff @ X_eff.T
        grad_W_mask = (active_grad_branches == 0) | self.consolidated_mask
        grad_W = RNullTensor(grad_W_val, grad_W_mask)

        active_WT = (~self.W.m.T).astype(np.float64)
        active_down = active_WT @ active_g
        W_eff_T = np.where(self.W.m.T, 0.0, self.W.v.T)
        grad_X = RNullTensor(W_eff_T @ g_eff, active_down == 0)

        return grad_X, grad_W


class NullMLP:
    """Multi-Layer Perceptron over R_null with LeakyReLU activation."""
    def __init__(self, dims: list, alpha: float = 0.01, seed: int = 42):
        self.layers = [
            NullLinear(dims[i], dims[i + 1], seed=seed + i)
            for i in range(len(dims) - 1)
        ]
        self.alpha = alpha
        self.z_cache = []

    def forward(self, X: RNullTensor) -> RNullTensor:
        self.z_cache = []
        curr = X
        for i, layer in enumerate(self.layers):
            z = layer.forward(curr)
            self.z_cache.append(z)
            if i < len(self.layers) - 1:
                act_v = np.where(z.v > 0.0, z.v, self.alpha * z.v)
                curr = RNullTensor(act_v, z.m.copy())
            else:
                curr = z
        return curr

    def backward(self, grad_loss: RNullTensor) -> list:
        grads_W = [None] * len(self.layers)
        delta = grad_loss
        for i in reversed(range(len(self.layers))):
            if i < len(self.layers) - 1:
                z = self.z_cache[i]
                d_phi = RNullTensor(np.where(z.v > 0.0, 1.0, self.alpha), z.m.copy())
                delta = causal_coupling(delta, d_phi)
            delta, grad_W = self.layers[i].backward(delta)
            grads_W[i] = grad_W
        return grads_W
