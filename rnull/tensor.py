"""NumPy reference implementation of the R_null algebra.

R_null is the set of real numbers extended with one extra element, written
``null`` here and ∅ in the documentation, which stands for "no value was
supplied".  An :class:`RNullArray` stores it as a pair of arrays:

* ``v`` -- the floating-point payload;
* ``m`` -- a boolean mask, ``True`` where the element is null.

The payload under the mask is always stored as ``0.0`` (canonical form), so a
value hidden behind the mask can never leak into a result.

Operator summary (``x`` real, ``n`` null):

==========================  ===========================  =====================
Operation                   Rule                         Result mask
==========================  ===========================  =====================
``a + b``  (oplus)          ``x + n = x``                ``a.m & b.m``
``a - b``  (ominus)         ``x - n = x``, ``n - x = -x``  ``a.m & b.m``
``a * b``  (otimes)         ``x * n = x``                ``a.m & b.m``
``a / b``  (oslash)         null if either side is null  ``a.m | b.m``
``w.synaptic_mul(x)``       null if either side is null  ``w.m | x.m``
``w.synaptic_skip(x)``      ``n |> x = x``, ``w |> n = n``  ``x.m``
==========================  ===========================  =====================

Division between two real operands follows IEEE-754 (``1/0 = inf``,
``0/0 = nan``); it is *not* redefined by the algebra.
"""

from __future__ import annotations

import numpy as np

__all__ = ["RNullArray", "normalization_exponent"]


def normalization_exponent(normalize) -> float:
    """Map a ``normalize`` option to the exponent ``a`` of the rescaling
    factor ``(fan_in / active) ** a`` applied to a partially observed sum.

    * ``False`` / ``None`` / ``"none"`` -> 0   (no rescaling)
    * ``True`` / ``"mean"``             -> 1   (unbiased for the full sum when
      entries are missing completely at random; the convention of inverted
      dropout and partial convolutions)
    * ``"variance"``                    -> 1/2 (keeps the second moment
      constant for independent zero-mean inputs)
    """
    if normalize is None or normalize is False or normalize == "none":
        return 0.0
    if normalize is True or normalize == "mean":
        return 1.0
    if normalize == "variance":
        return 0.5
    raise ValueError(
        f"normalize must be False, True, 'none', 'mean' or 'variance'; got {normalize!r}"
    )


def _as_float_array(values, dtype=None) -> np.ndarray:
    """Return a writable floating-point copy of ``values``."""
    arr = np.asarray(values)
    if dtype is None:
        dtype = arr.dtype if np.issubdtype(arr.dtype, np.floating) else np.float64
    return np.array(arr, dtype=dtype)


class RNullArray:
    """Array of elements of R_null, stored as a (payload, mask) pair.

    Parameters
    ----------
    v:
        Payload values (array-like, scalar, or another :class:`RNullArray`).
        Floating dtypes are preserved; anything else is converted to float64.
    is_null:
        Boolean mask broadcastable to ``v``; ``True`` marks a null element.
        Defaults to all ``False`` (every element is a real number).
    dtype:
        Optional floating dtype for the payload.
    """

    __slots__ = ("v", "m")

    # Make ``ndarray <op> RNullArray`` defer to the reflected operators below
    # instead of building an object array.
    __array_ufunc__ = None

    def __init__(self, v, is_null=None, dtype=None):
        if isinstance(v, RNullArray):
            if is_null is None:
                is_null = v.m
            v = v.v
        self.v = _as_float_array(v, dtype)
        if is_null is None:
            self.m = np.zeros(self.v.shape, dtype=bool)
        else:
            mask = np.asarray(is_null, dtype=bool)
            self.m = np.array(np.broadcast_to(mask, self.v.shape))
        np.putmask(self.v, self.m, 0.0)

    # ------------------------------------------------------------------
    # Constructors and basic protocol
    # ------------------------------------------------------------------
    @classmethod
    def full_null(cls, shape, dtype=np.float64) -> "RNullArray":
        """Array of the given shape in which every element is null."""
        return cls(np.zeros(shape, dtype=dtype), np.ones(shape, dtype=bool))

    @classmethod
    def null_like(cls, other) -> "RNullArray":
        """All-null array with the shape and dtype of ``other``."""
        ref = other.v if isinstance(other, RNullArray) else _as_float_array(other)
        return cls.full_null(ref.shape, dtype=ref.dtype)

    @classmethod
    def constant(cls, value: float, shape, dtype=np.float64) -> "RNullArray":
        """Real-valued array filled with ``value``."""
        return cls(np.full(shape, value, dtype=dtype))

    @property
    def shape(self):
        return self.v.shape

    @property
    def ndim(self) -> int:
        return self.v.ndim

    @property
    def size(self) -> int:
        return self.v.size

    @property
    def dtype(self):
        return self.v.dtype

    @property
    def T(self) -> "RNullArray":
        return RNullArray(self.v.T, self.m.T)

    def __len__(self) -> int:
        return len(self.v)

    def __getitem__(self, index) -> "RNullArray":
        return RNullArray(self.v[index], self.m[index])

    def copy(self) -> "RNullArray":
        return RNullArray(self.v, self.m)

    def __repr__(self) -> str:
        def render(value, null):
            if isinstance(value, list):
                return [render(v, n) for v, n in zip(value, null)]
            return "∅" if null else value

        return f"RNullArray({render(np.round(self.v, 4).tolist(), self.m.tolist())!r})"

    def _coerce(self, other) -> "RNullArray":
        if isinstance(other, RNullArray):
            return other
        if np.isscalar(other):
            return RNullArray(other, dtype=self.v.dtype)
        return RNullArray(other)

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------
    def filled(self, value: float = 0.0) -> np.ndarray:
        """Plain ndarray with null elements replaced by ``value``."""
        return np.where(self.m, np.asarray(value, dtype=self.v.dtype), self.v)

    def count_present(self, axis=None):
        """Number of non-null elements (optionally along ``axis``)."""
        return np.sum(~self.m, axis=axis)

    def equals(self, other, rtol: float = 0.0, atol: float = 0.0) -> bool:
        """True if both arrays have the same mask and the same live values."""
        other = self._coerce(other)
        if self.shape != other.shape or not np.array_equal(self.m, other.m):
            return False
        return bool(np.allclose(self.v, other.v, rtol=rtol, atol=atol, equal_nan=True))

    # ------------------------------------------------------------------
    # oplus / ominus: null is the additive identity
    # ------------------------------------------------------------------
    def oplus(self, other) -> "RNullArray":
        other = self._coerce(other)
        return RNullArray(self.v + other.v, self.m & other.m)

    def __neg__(self) -> "RNullArray":
        return RNullArray(-self.v, self.m)

    def ominus(self, other) -> "RNullArray":
        return self.oplus(-self._coerce(other))

    __add__ = oplus
    __radd__ = oplus
    __sub__ = ominus

    def __rsub__(self, other) -> "RNullArray":
        return self._coerce(other).ominus(self)

    # ------------------------------------------------------------------
    # otimes: null is also the multiplicative identity
    # ------------------------------------------------------------------
    def otimes(self, other) -> "RNullArray":
        other = self._coerce(other)
        a = np.where(self.m, 1.0, self.v)
        b = np.where(other.m, 1.0, other.v)
        return RNullArray(a * b, self.m & other.m)

    __mul__ = otimes
    __rmul__ = otimes

    # ------------------------------------------------------------------
    # oslash: null absorbs; real/real follows IEEE-754
    # ------------------------------------------------------------------
    def oslash(self, other) -> "RNullArray":
        other = self._coerce(other)
        mask = self.m | other.m
        with np.errstate(divide="ignore", invalid="ignore"):
            quotient = self.v / np.where(mask, 1.0, other.v)
        return RNullArray(quotient, mask)

    __truediv__ = oslash

    def __rtruediv__(self, other) -> "RNullArray":
        return self._coerce(other).oslash(self)

    # ------------------------------------------------------------------
    # Synaptic couplings between a weight (self) and an input (x)
    # ------------------------------------------------------------------
    def synaptic_mul(self, x) -> "RNullArray":
        """Absorbing coupling ``w (.) x``: null if the weight or the input is null.

        Summing the results with ``+`` skips the null synapses, which is the
        behaviour of the ``rnull_mac`` hardware cell.
        """
        x = self._coerce(x)
        return RNullArray(self.v * x.v, self.m | x.m)

    def synaptic_skip(self, x) -> "RNullArray":
        """Directed coupling ``w |> x``.

        A null weight passes a real input through unchanged (weight of one);
        a null input makes the result null.  Note that this is *not* pruning:
        a pruned synapse should contribute nothing, which is what
        :meth:`synaptic_mul` does.
        """
        x = self._coerce(x)
        out = np.where(self.m, 1.0, self.v) * x.v
        return RNullArray(out, np.broadcast_to(x.m, out.shape))

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------
    def sum(self, axis=None, keepdims: bool = False) -> "RNullArray":
        """oplus-reduction: nulls are skipped; null only if every element is null."""
        return RNullArray(
            self.v.sum(axis=axis, keepdims=keepdims),
            self.m.all(axis=axis, keepdims=keepdims),
        )

    def prod(self, axis=None, keepdims: bool = False) -> "RNullArray":
        """otimes-reduction: nulls are skipped; null only if every element is null."""
        return RNullArray(
            np.where(self.m, 1.0, self.v).prod(axis=axis, keepdims=keepdims),
            self.m.all(axis=axis, keepdims=keepdims),
        )

    # ------------------------------------------------------------------
    # Null detector
    # ------------------------------------------------------------------
    def chi_null(self, evaluate_polynomial: bool = False) -> np.ndarray:
        """Indicator of null elements: 1.0 where null, 0.0 where real.

        By default this simply reads the mask.  With
        ``evaluate_polynomial=True`` it evaluates

            ((1 * 0) + (1 * x)) - (1 * (0 + x))

        using the algebra's own operators.  The expression exploits the
        failure of distributivity around null and agrees with the mask for
        every *finite* real ``x``.  For ``x = ±inf`` or ``nan`` it returns
        ``nan`` (``inf - inf``), so the mask is the authoritative detector.
        """
        if not evaluate_polynomial:
            return self.m.astype(self.v.dtype)
        one = RNullArray(np.ones_like(self.v))
        zero = RNullArray(np.zeros_like(self.v))
        with np.errstate(invalid="ignore"):
            return (((one * zero) + (one * self)) - (one * (zero + self))).v
