"""
Core Tensor Engine for Causal Transparency Algebra (R_null).
Author: Hakim Guettal
"""

import numpy as np


class RNullTensor:
    """
    Vectorized Dual-Tensor representation of elements in R_null = R U {null}.
    
    Attributes:
        v (np.ndarray): Metric real values (float64).
        m (np.ndarray): Topological transparency bitmask (bool).
                        True indicates the Universal Transparency Operator (null).
    """
    __slots__ = ("v", "m")

    def __init__(self, v, m=None):
        self.v = np.asarray(v, dtype=np.float64)
        if m is None:
            self.m = np.zeros_like(self.v, dtype=bool)
        else:
            self.m = np.broadcast_to(np.asarray(m, dtype=bool), self.v.shape).copy()

    @classmethod
    def full_null(cls, shape):
        """Creates a tensor completely filled with the transparency operator (null)."""
        return cls(np.zeros(shape, dtype=np.float64), np.ones(shape, dtype=bool))

    @classmethod
    def constant(cls, val: float, shape):
        """Creates a real tensor filled with a constant metric value."""
        return cls(np.full(shape, val, dtype=np.float64), np.zeros(shape, dtype=bool))

    @property
    def shape(self):
        return self.v.shape

    @property
    def size(self):
        return self.v.size

    def copy(self) -> "RNullTensor":
        return RNullTensor(self.v.copy(), self.m.copy())

    def __repr__(self):
        out = np.round(self.v, 4).astype(object)
        out[self.m] = "∅"
        return f"RNullTensor({out})"

    # Axiom 1a: Bilateral Additive Identity (x (+) null = null (+) x = x)
    def oplus(self, other: "RNullTensor") -> "RNullTensor":
        v_self = np.where(self.m, 0.0, self.v)
        v_other = np.where(other.m, 0.0, other.v)
        return RNullTensor(v_self + v_other, self.m & other.m)

    def __add__(self, other: "RNullTensor") -> "RNullTensor":
        return self.oplus(other)

    # Axiom 3: A-Polarity (-null = null)
    def __neg__(self) -> "RNullTensor":
        return RNullTensor(-self.v, self.m.copy())

    def ominus(self, other: "RNullTensor") -> "RNullTensor":
        return self.oplus(-other)

    def __sub__(self, other: "RNullTensor") -> "RNullTensor":
        return self.ominus(other)

    # Axiom 1b: Bilateral Multiplicative Identity (x (*) null = null (*) x = x, 0 (*) null = 0)
    def otimes(self, other: "RNullTensor") -> "RNullTensor":
        v_self = np.where(self.m, 1.0, self.v)
        v_other = np.where(other.m, 1.0, other.v)
        return RNullTensor(v_self * v_other, self.m & other.m)

    def __mul__(self, other: "RNullTensor") -> "RNullTensor":
        return self.otimes(other)

    # Axiom 2: Quotient Collapse (x (/) null = null, null (/) x = null, null (/) 0 = null)
    def oslash(self, other: "RNullTensor") -> "RNullTensor":
        m_out = self.m | other.m
        safe_denom = np.where(m_out | (other.v == 0.0), 1.0, other.v)
        v_out = self.v / safe_denom
        return RNullTensor(v_out, m_out)

    def __truediv__(self, other: "RNullTensor") -> "RNullTensor":
        return self.oslash(other)

    def causal_defect_chi(self) -> "RNullTensor":
        """
        Algebraic Transparency Detector polynomial:
        chi(x) = ((1 (*) 0) (+) (1 (*) x)) (-) (1 (*) (0 (+) x))
        Evaluates strictly to 0.0 for all x in R (including 0.0), and 1.0 for x = null.
        """
        one = RNullTensor.constant(1.0, self.v.shape)
        zero = RNullTensor.constant(0.0, self.v.shape)
        late_fusion = (one * zero) + (one * self)
        early_fusion = one * (zero + self)
        return late_fusion - early_fusion


def causal_coupling(w: RNullTensor, x: RNullTensor) -> RNullTensor:
    """
    Causal Synaptic Coupling Operator (w |> x):
    Evaluates to w * x when both are in R, and collapses to null if either is null.
    Derived algebraically via Axiom 2 quotient collapse.
    """
    return RNullTensor(w.v * x.v, w.m | x.m)
