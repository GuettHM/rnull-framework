"""NumPy reference layers over R_null (forward and manual backward passes).

All layers are batch-first: inputs have shape ``(batch, in_features)``.
These classes exist to make the semantics explicit and testable without a
deep-learning framework; :mod:`rnull.core` provides the PyTorch equivalents.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .tensor import RNullArray, normalization_exponent

__all__ = [
    "fuse_product",
    "present_count",
    "MultimodalProductFusion",
    "NullLinear",
    "NullMLP",
]

ParamSlot = Tuple[object, str]


def fuse_product(modalities: Sequence[RNullArray]) -> RNullArray:
    """Element-wise otimes-product of several modalities.

    Null modalities are skipped; the result is null only where *all* of them
    are null.  The magnitude of the result depends on how many modalities are
    present, and the result mask does not record partial absence -- use
    :func:`present_count` alongside it when downstream code needs to know.
    """
    if not modalities:
        raise ValueError("modalities must not be empty")
    out = modalities[0].copy()
    for modality in modalities[1:]:
        out = out.otimes(modality)
    return out


def present_count(modalities: Sequence[RNullArray]) -> np.ndarray:
    """Element-wise number of non-null modalities."""
    if not modalities:
        raise ValueError("modalities must not be empty")
    masks = np.broadcast_arrays(*[m.m for m in modalities])
    return np.sum([~mask for mask in masks], axis=0)


class MultimodalProductFusion:
    """Thin layer wrapper around :func:`fuse_product`."""

    def forward(self, modalities: Sequence[RNullArray]) -> RNullArray:
        return fuse_product(modalities)

    __call__ = forward


class NullLinear:
    """Linear layer ``y_i = sum_j (W_ij (.) x_j) + b_i`` over R_null.

    * A synapse is skipped when its weight or its input is null.
    * An output is null (and its bias is suppressed) when *every* synapse
      feeding it was skipped.
    * ``layer.last_coverage`` holds, per output, the fraction of non-pruned
      synapses that had a live input, so partial absence is observable.

    With ``normalize=False`` the numerical output for a partially missing
    input is identical to an ordinary linear layer applied to the
    zero-imputed input; the only additions are the all-missing flag and the
    coverage.  Otherwise the weighted sum over the ``k`` active synapses of a
    neuron with ``n`` non-pruned weights is multiplied by ``(n / k) ** a``:

    * ``normalize="mean"`` (or ``True``), ``a = 1``: an unbiased estimate of
      the full sum when inputs are missing completely at random (the
      convention of inverted dropout and partial convolutions).  It inflates
      the variance by ``n / k`` for independent zero-mean inputs.
    * ``normalize="variance"``, ``a = 1/2``: keeps the second moment of the
      sum constant for independent zero-mean inputs.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        normalize=False,
        seed: int = 0,
    ):
        normalization_exponent(normalize)  # validate early
        rng = np.random.default_rng(seed)
        std = np.sqrt(2.0 / (in_features + out_features))
        self.in_features = in_features
        self.out_features = out_features
        self.normalize = normalize
        self.W = RNullArray(rng.normal(0.0, std, (out_features, in_features)))
        self.b: Optional[RNullArray] = RNullArray(np.zeros(out_features)) if bias else None
        # Entries set to True never receive a gradient (their gradient is null).
        self.consolidated_mask = np.zeros((out_features, in_features), dtype=bool)
        self.last_coverage: Optional[np.ndarray] = None
        self.last_metrics: dict = {}
        self._cache = None

    def parameters(self) -> List[ParamSlot]:
        slots: List[ParamSlot] = [(self, "W")]
        if self.b is not None:
            slots.append((self, "b"))
        return slots

    def forward(self, X) -> RNullArray:
        X = X if isinstance(X, RNullArray) else RNullArray(X)
        if X.ndim != 2 or X.shape[1] != self.in_features:
            raise ValueError(
                f"expected input of shape (batch, {self.in_features}), got {X.shape}"
            )
        dtype = self.W.dtype
        live_w = (~self.W.m).astype(dtype)            # (out, in)
        live_x = (~X.m).astype(dtype)                 # (batch, in)
        n_active = live_x @ live_w.T                  # (batch, out)
        fan_in = live_w.sum(axis=1)                   # (out,)

        scale = np.ones_like(n_active)
        exponent = normalization_exponent(self.normalize)
        if exponent:
            scale = (fan_in / np.maximum(n_active, 1.0)) ** exponent

        # Payloads are 0 under the mask, so null synapses contribute nothing.
        z = (X.v @ self.W.v.T) * scale
        out_mask = n_active == 0
        if self.b is not None:
            z = z + self.b.filled(0.0)
        out = RNullArray(z, out_mask)

        total = X.shape[0] * self.W.size
        executed = int(n_active.sum())
        self.last_coverage = n_active / np.maximum(fan_in, 1.0)
        self.last_metrics = {
            "total_macs": total,
            "executed_macs": executed,
            "skipped_macs": total - executed,
            "skipped_pct": 100.0 * (total - executed) / total if total else 0.0,
        }
        self._cache = (X, scale, out_mask)
        return out

    __call__ = forward

    def backward(self, grad_out: RNullArray):
        """Back-propagate ``dL/dy``; returns ``(grad_X, grad_W, grad_b)``.

        A gradient entry is null when no live (output, input) pair
        contributed to it, when the corresponding weight is null, or when the
        weight is marked in ``consolidated_mask``.
        """
        if self._cache is None:
            raise RuntimeError("backward() called before forward()")
        X, scale, out_mask = self._cache
        grad_out = grad_out if isinstance(grad_out, RNullArray) else RNullArray(grad_out)

        g_mask = grad_out.m | out_mask
        g = np.where(g_mask, 0.0, grad_out.v)
        g_scaled = g * scale
        dtype = self.W.dtype
        live_g = (~g_mask).astype(dtype)              # (batch, out)
        live_x = (~X.m).astype(dtype)                 # (batch, in)
        live_w = (~self.W.m).astype(dtype)            # (out, in)

        pair_count = live_g.T @ live_x                # (out, in)
        grad_W = RNullArray(
            g_scaled.T @ X.v,
            (pair_count == 0) | self.W.m | self.consolidated_mask,
        )
        grad_b = None
        if self.b is not None:
            grad_b = RNullArray(g.sum(axis=0), live_g.sum(axis=0) == 0)
        grad_X = RNullArray(g_scaled @ self.W.v, ((live_g @ live_w) == 0) | X.m)
        return grad_X, grad_W, grad_b


class NullMLP:
    """Multi-layer perceptron over R_null with LeakyReLU activations."""

    def __init__(
        self,
        dims: Sequence[int],
        alpha: float = 0.01,
        seed: int = 0,
        normalize=False,
    ):
        if len(dims) < 2:
            raise ValueError("dims must contain at least an input and an output size")
        self.layers = [
            NullLinear(dims[i], dims[i + 1], normalize=normalize, seed=seed + i)
            for i in range(len(dims) - 1)
        ]
        self.alpha = alpha
        self._z_cache: List[RNullArray] = []

    def parameters(self) -> List[ParamSlot]:
        return [slot for layer in self.layers for slot in layer.parameters()]

    def forward(self, X) -> RNullArray:
        self._z_cache = []
        current = X
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            z = layer.forward(current)
            self._z_cache.append(z)
            if i < last:
                current = RNullArray(np.where(z.v > 0.0, z.v, self.alpha * z.v), z.m)
            else:
                current = z
        return current

    __call__ = forward

    def backward(self, grad_loss: RNullArray) -> List[Optional[RNullArray]]:
        """Return gradients aligned with :meth:`parameters`."""
        per_layer: List[Tuple[RNullArray, Optional[RNullArray]]] = [None] * len(self.layers)
        delta = grad_loss if isinstance(grad_loss, RNullArray) else RNullArray(grad_loss)
        last = len(self.layers) - 1
        for i in reversed(range(len(self.layers))):
            if i < last:
                z = self._z_cache[i]
                slope = np.where(z.v > 0.0, 1.0, self.alpha)
                delta = RNullArray(delta.v * slope, delta.m | z.m)
            delta, grad_W, grad_b = self.layers[i].backward(delta)
            per_layer[i] = (grad_W, grad_b)
        grads: List[Optional[RNullArray]] = []
        for layer, (grad_W, grad_b) in zip(self.layers, per_layer):
            grads.append(grad_W)
            if layer.b is not None:
                grads.append(grad_b)
        return grads
