"""NumPy reference optimisers that treat a null gradient as "do not touch".

Each optimiser works on *parameter slots* -- ``(owner, attribute_name)``
pairs whose attribute is an :class:`~rnull.tensor.RNullArray` -- and on a
list of gradients aligned with those slots (see ``NullMLP.parameters()`` and
``NullMLP.backward()``).

Where a gradient element is null, the corresponding parameter element is left
bit-for-bit unchanged: no gradient step, no momentum, no weight decay, and
its optimiser state is not advanced.  This is a masked update expressed
through the algebra (``w + null = w``); it produces exactly the same numbers
as ``w[~mask] += delta[~mask]``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .tensor import RNullArray

__all__ = ["NullSGD", "NullAdamW", "NullAdam"]

ParamSlot = Tuple[object, str]


class _Optimizer:
    def __init__(self, params: Sequence[ParamSlot], no_decay: Sequence[str] = ("b",)):
        self.params: List[ParamSlot] = list(params)
        if not self.params:
            raise ValueError("optimizer received an empty parameter list")
        self.no_decay = tuple(no_decay)

    @staticmethod
    def _get(slot: ParamSlot) -> RNullArray:
        owner, name = slot
        return getattr(owner, name)

    @staticmethod
    def _set(slot: ParamSlot, value: RNullArray) -> None:
        owner, name = slot
        setattr(owner, name, value)

    def _decay_for(self, slot: ParamSlot, weight_decay: float) -> float:
        return 0.0 if slot[1] in self.no_decay else weight_decay

    def _check(self, grads: Sequence[Optional[RNullArray]]) -> None:
        if len(grads) != len(self.params):
            raise ValueError(
                f"expected {len(self.params)} gradients, got {len(grads)}"
            )


class NullSGD(_Optimizer):
    """SGD with momentum and (coupled, L2) weight decay."""

    def __init__(
        self,
        params: Sequence[ParamSlot],
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        no_decay: Sequence[str] = ("b",),
    ):
        super().__init__(params, no_decay)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity = [np.zeros(self._get(s).shape) for s in self.params]

    def step(self, grads: Sequence[Optional[RNullArray]]) -> None:
        self._check(grads)
        for i, (slot, grad) in enumerate(zip(self.params, grads)):
            if grad is None:
                continue
            param = self._get(slot)
            # A null parameter stays null; a null gradient freezes the entry.
            active = ~grad.m & ~param.m
            effective = grad.v + self._decay_for(slot, self.weight_decay) * param.v
            self._velocity[i] = np.where(
                active, self.momentum * self._velocity[i] + effective, self._velocity[i]
            )
            delta = RNullArray(-self.lr * self._velocity[i], ~active)
            self._set(slot, param.oplus(delta))


class NullAdamW(_Optimizer):
    """Adam with decoupled weight decay (AdamW).

    Bias correction uses a per-element step count, so an element that was
    frozen for a while resumes with correctly scaled moment estimates.
    """

    def __init__(
        self,
        params: Sequence[ParamSlot],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        no_decay: Sequence[str] = ("b",),
    ):
        super().__init__(params, no_decay)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        shapes = [self._get(s).shape for s in self.params]
        self._m = [np.zeros(shape) for shape in shapes]
        self._v = [np.zeros(shape) for shape in shapes]
        self._t = [np.zeros(shape) for shape in shapes]

    def step(self, grads: Sequence[Optional[RNullArray]]) -> None:
        self._check(grads)
        for i, (slot, grad) in enumerate(zip(self.params, grads)):
            if grad is None:
                continue
            param = self._get(slot)
            active = ~grad.m & ~param.m
            g = grad.v

            self._t[i] = np.where(active, self._t[i] + 1.0, self._t[i])
            self._m[i] = np.where(
                active, self.beta1 * self._m[i] + (1.0 - self.beta1) * g, self._m[i]
            )
            self._v[i] = np.where(
                active, self.beta2 * self._v[i] + (1.0 - self.beta2) * g * g, self._v[i]
            )

            t = np.maximum(self._t[i], 1.0)
            m_hat = self._m[i] / (1.0 - self.beta1 ** t)
            v_hat = self._v[i] / (1.0 - self.beta2 ** t)
            decay = self._decay_for(slot, self.weight_decay)
            step = -self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + decay * param.v)
            self._set(slot, param.oplus(RNullArray(step, ~active)))


#: Backwards-compatible alias.
NullAdam = NullAdamW
