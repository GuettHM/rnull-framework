"""PyTorch implementation of the R_null algebra.

Requires PyTorch (``pip install "rnull-framework[torch]"``).  The semantics
are identical to the NumPy reference in :mod:`rnull.tensor`; see that module
for the operator table.  All operations are differentiable with respect to
the payload of the non-null elements.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn

from .tensor import normalization_exponent

__all__ = [
    "RNullTensor",
    "RNullLinear",
    "continual_learning_step",
    "protected_step",
]


def _payload(data, like: "torch.Tensor | None" = None) -> torch.Tensor:
    """Convert ``data`` to a floating-point tensor without copying tensors."""
    if isinstance(data, torch.Tensor):
        value = data
    elif like is not None:
        value = torch.as_tensor(data, dtype=like.dtype, device=like.device)
    else:
        value = torch.as_tensor(data)
    if not value.is_floating_point():
        value = value.to(torch.get_default_dtype())
    return value


class RNullTensor:
    """Tensor of elements of R_null, stored as a (payload, mask) pair.

    ``v`` is the floating-point payload and ``m`` a boolean mask that is
    ``True`` where the element is null.  The payload under the mask is kept
    at ``0.0`` so hidden values cannot leak into results.  Floating dtypes of
    the input are preserved.
    """

    def __init__(self, data, is_null=None):
        if isinstance(data, RNullTensor):
            if is_null is None:
                is_null = data.m
            data = data.v
        value = _payload(data)
        if is_null is None:
            mask = torch.zeros_like(value, dtype=torch.bool)
        else:
            mask = (
                torch.as_tensor(is_null, dtype=torch.bool, device=value.device)
                .expand_as(value)
                .clone()
            )
        self.v = value.masked_fill(mask, 0.0)
        self.m = mask

    # ------------------------------------------------------------------
    # Constructors and basic protocol
    # ------------------------------------------------------------------
    @classmethod
    def null_like(cls, tensor) -> "RNullTensor":
        """All-null tensor with the shape, dtype and device of ``tensor``."""
        ref = tensor.v if isinstance(tensor, RNullTensor) else _payload(tensor)
        return cls(torch.zeros_like(ref), is_null=torch.ones_like(ref, dtype=torch.bool))

    @property
    def shape(self):
        return self.v.shape

    @property
    def dtype(self):
        return self.v.dtype

    @property
    def device(self):
        return self.v.device

    def __getitem__(self, index) -> "RNullTensor":
        return RNullTensor(self.v[index], is_null=self.m[index])

    def __repr__(self) -> str:
        return f"RNullTensor(v={self.v}, is_null={self.m})"

    def _coerce(self, other) -> "RNullTensor":
        if isinstance(other, RNullTensor):
            return other
        return RNullTensor(_payload(other, like=self.v))

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------
    def filled(self, value: float = 0.0) -> torch.Tensor:
        """Plain tensor with null elements replaced by ``value``."""
        return self.v.masked_fill(self.m, value)

    def count_present(self, dim=None) -> torch.Tensor:
        """Number of non-null elements (optionally along ``dim``)."""
        live = (~self.m).to(torch.long)
        return live.sum() if dim is None else live.sum(dim=dim)

    # ------------------------------------------------------------------
    # oplus / ominus: null is the additive identity
    # ------------------------------------------------------------------
    def __add__(self, other) -> "RNullTensor":
        other = self._coerce(other)
        return RNullTensor(self.v + other.v, is_null=self.m & other.m)

    __radd__ = __add__

    def __neg__(self) -> "RNullTensor":
        return RNullTensor(-self.v, is_null=self.m)

    def __sub__(self, other) -> "RNullTensor":
        return self.__add__(-self._coerce(other))

    def __rsub__(self, other) -> "RNullTensor":
        return self._coerce(other).__sub__(self)

    # ------------------------------------------------------------------
    # otimes: null is also the multiplicative identity
    # ------------------------------------------------------------------
    def __mul__(self, other) -> "RNullTensor":
        other = self._coerce(other)
        product = self.v.masked_fill(self.m, 1.0) * other.v.masked_fill(other.m, 1.0)
        return RNullTensor(product, is_null=self.m & other.m)

    __rmul__ = __mul__

    # ------------------------------------------------------------------
    # oslash: null absorbs; real/real follows IEEE-754
    # ------------------------------------------------------------------
    def __truediv__(self, other) -> "RNullTensor":
        other = self._coerce(other)
        quotient = self.v / other.v.masked_fill(other.m, 1.0)
        return RNullTensor(quotient, is_null=self.m | other.m)

    def __rtruediv__(self, other) -> "RNullTensor":
        return self._coerce(other).__truediv__(self)

    # ------------------------------------------------------------------
    # Synaptic couplings between a weight (self) and an input (x)
    # ------------------------------------------------------------------
    def synaptic_mul(self, x) -> "RNullTensor":
        """Absorbing coupling ``w (.) x``: null if the weight or the input is null."""
        x = self._coerce(x)
        return RNullTensor(self.v * x.v, is_null=self.m | x.m)

    def synaptic_skip(self, x) -> "RNullTensor":
        """Directed coupling ``w |> x``: a null weight passes a real input
        through unchanged; a null input makes the result null.  This is not
        pruning -- use :meth:`synaptic_mul` for synapses that should
        contribute nothing."""
        x = self._coerce(x)
        return RNullTensor(self.v.masked_fill(self.m, 1.0) * x.v, is_null=x.m)

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------
    def sum(self, dim=None) -> "RNullTensor":
        """oplus-reduction: nulls are skipped; null only if all elements are null."""
        if dim is None:
            return RNullTensor(self.v.sum(), is_null=self.m.all())
        return RNullTensor(self.v.sum(dim=dim), is_null=self.m.all(dim=dim))

    def prod(self, dim=None) -> "RNullTensor":
        """otimes-reduction: nulls are skipped; null only if all elements are null."""
        ones = self.v.masked_fill(self.m, 1.0)
        if dim is None:
            return RNullTensor(ones.prod(), is_null=self.m.all())
        return RNullTensor(ones.prod(dim=dim), is_null=self.m.all(dim=dim))

    # ------------------------------------------------------------------
    # Null detector
    # ------------------------------------------------------------------
    def chi_null(self, evaluate_polynomial: bool = False) -> torch.Tensor:
        """Indicator of null elements: 1.0 where null, 0.0 where real.

        By default this reads the mask.  With ``evaluate_polynomial=True`` it
        evaluates ``((1 * 0) + (1 * x)) - (1 * (0 + x))`` with the algebra's
        own operators; the two agree for every finite real ``x``.  For
        ``±inf`` or ``nan`` the polynomial returns ``nan``, so the mask is
        the authoritative detector.
        """
        if not evaluate_polynomial:
            return self.m.to(self.v.dtype)
        one = RNullTensor(torch.ones_like(self.v))
        zero = RNullTensor(torch.zeros_like(self.v))
        return (((one * zero) + (one * self)) - (one * (zero + self))).v


class RNullLinear(nn.Module):
    """Linear layer ``y_i = sum_j (W_ij (.) x_j) + b_i`` over R_null.

    * A synapse is skipped when its weight is marked null in
      ``weight_null_mask`` or its input is null.
    * An output is null, and its bias suppressed, when every synapse feeding
      it was skipped (for example when the whole input is missing).
    * ``forward(x, return_coverage=True)`` also returns, per output, the
      fraction of non-pruned synapses that had a live input.

    With ``normalize=False`` the numerical output for a partially missing
    input equals an ordinary ``nn.Linear`` applied to the zero-imputed input.
    Otherwise the sum over the ``k`` active synapses of a neuron with ``n``
    non-pruned weights is multiplied by ``(n / k) ** a``: ``a = 1`` for
    ``normalize="mean"`` (or ``True``; unbiased for the full sum under
    completely-at-random missingness) and ``a = 1/2`` for
    ``normalize="variance"`` (constant second moment for independent
    zero-mean inputs).

    Inputs may have any leading batch dimensions: ``(..., in_features)``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        normalize=False,
    ):
        super().__init__()
        normalization_exponent(normalize)  # validate early
        self.in_features = in_features
        self.out_features = out_features
        self.normalize = normalize
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * (1.0 / (in_features ** 0.5))
        )
        self.register_buffer(
            "weight_null_mask", torch.zeros(out_features, in_features, dtype=torch.bool)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def set_weight_null_mask(self, mask) -> None:
        """Mark weights as null (``True``) so their synapses are skipped."""
        mask = torch.as_tensor(mask, dtype=torch.bool, device=self.weight.device)
        if mask.shape != self.weight.shape:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} does not match weight shape "
                f"{tuple(self.weight.shape)}"
            )
        self.weight_null_mask = mask.clone()

    def forward(self, x, return_coverage: bool = False):
        if not isinstance(x, RNullTensor):
            x = RNullTensor(x)
        dtype = self.weight.dtype
        weight = self.weight.masked_fill(self.weight_null_mask, 0.0)
        live_w = (~self.weight_null_mask).to(dtype)        # [out, in]
        live_x = (~x.m).to(dtype)                          # [..., in]
        n_active = live_x @ live_w.t()                     # [..., out]
        fan_in = live_w.sum(dim=-1)                        # [out]

        # Payloads are 0 under the mask, so null synapses contribute nothing.
        out_v = x.v.to(dtype) @ weight.t()
        exponent = normalization_exponent(self.normalize)
        if exponent:
            out_v = out_v * (fan_in / n_active.clamp(min=1.0)) ** exponent
        out_m = n_active == 0
        if self.bias is not None:
            out_v = out_v + self.bias
        out = RNullTensor(out_v, is_null=out_m)
        if return_coverage:
            return out, n_active / fan_in.clamp(min=1.0)
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, normalize={self.normalize}"
        )


@torch.no_grad()
def continual_learning_step(param, raw_delta, frozen_mask) -> RNullTensor:
    """Apply ``param <- param + delta`` with ``delta`` null where frozen.

    Entries selected by ``frozen_mask`` are left bit-for-bit unchanged.  This
    is a masked update written through the algebra; it is numerically
    identical to ``param[~frozen_mask] += raw_delta[~frozen_mask]``.
    """
    current = RNullTensor(param.data)
    delta = RNullTensor(raw_delta, is_null=frozen_mask)
    updated = current + delta
    param.data.copy_(updated.v)
    return updated


def protected_step(
    optimizer,
    protected: Iterable[Tuple["torch.Tensor", "torch.Tensor"]],
    closure=None,
):
    """Run ``optimizer.step()`` while keeping selected entries unchanged.

    ``protected`` is an iterable of ``(parameter, frozen_mask)`` pairs.  The
    frozen entries are snapshotted, the optimiser runs normally (including
    weight decay and momentum), and the snapshot is written back afterwards.
    Use this to freeze individual *elements* of a parameter with any stock
    optimiser such as ``torch.optim.AdamW``; to freeze a whole parameter,
    ``requires_grad_(False)`` is simpler.
    """
    snapshots = [
        (
            param,
            torch.as_tensor(mask, dtype=torch.bool, device=param.device),
            param.detach().clone(),
        )
        for param, mask in protected
    ]
    loss = optimizer.step(closure) if closure is not None else optimizer.step()
    with torch.no_grad():
        for param, mask, snapshot in snapshots:
            param.copy_(torch.where(mask, snapshot, param))
    return loss
