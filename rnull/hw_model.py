"""Bit-accurate Python reference model of ``hardware/rnull_mac.v``.

The model reproduces the register behaviour of the RTL cycle by cycle
(two's-complement wrap-around, sticky overflow flag, null tag) so that the
hardware semantics can be tested against the software algebra without a
Verilog simulator.  It does not model timing, area or power.
"""

from __future__ import annotations

from typing import Iterable

__all__ = ["RNullMacModel"]


def _wrap_signed(value: int, bits: int) -> int:
    """Interpret ``value`` modulo 2**bits as a signed two's-complement integer."""
    half = 1 << (bits - 1)
    return ((value + half) % (1 << bits)) - half


class RNullMacModel:
    """Multiply-accumulate cell with a 1-bit null tag per operand.

    Parameters mirror the RTL: operands are signed ``width``-bit integers and
    the accumulator is ``2 * width + guard`` bits wide.
    """

    def __init__(self, width: int = 16, guard: int = 8):
        if width < 2:
            raise ValueError("width must be at least 2")
        if guard < 1:
            raise ValueError("guard must be at least 1")
        self.width = width
        self.guard = guard
        self.acc_width = 2 * width + guard
        self.reset()

    def reset(self) -> None:
        """Asynchronous reset: accumulator null, counters cleared."""
        self.acc_val = 0
        self.acc_null = True
        self.acc_ovf = False
        self.cycles = 0
        self.active_cycles = 0

    def clear(self) -> None:
        """Synchronous ``clear_acc``: one clock cycle that empties the accumulator."""
        self.acc_val = 0
        self.acc_null = True
        self.acc_ovf = False
        self.cycles += 1

    def _check_operand(self, name: str, value: int) -> None:
        low, high = -(1 << (self.width - 1)), (1 << (self.width - 1)) - 1
        if not low <= value <= high:
            raise ValueError(f"{name}={value} does not fit in {self.width} signed bits")

    def step(self, w_val: int, w_null: bool, x_val: int, x_null: bool) -> None:
        """Advance one clock cycle."""
        self._check_operand("w_val", w_val)
        self._check_operand("x_val", x_val)
        self.cycles += 1
        if w_null or x_null:
            return  # clock enable low: the registers hold their state
        self.active_cycles += 1
        base = 0 if self.acc_null else self.acc_val
        exact = base + w_val * x_val
        wrapped = _wrap_signed(exact, self.acc_width)
        self.acc_ovf = self.acc_ovf or (wrapped != exact)
        self.acc_val = wrapped
        self.acc_null = False

    def run(
        self,
        w_vals: Iterable[int],
        w_nulls: Iterable[bool],
        x_vals: Iterable[int],
        x_nulls: Iterable[bool],
    ) -> "RNullMacModel":
        """Feed a sequence of operands, one per clock cycle."""
        for w, wn, x, xn in zip(w_vals, w_nulls, x_vals, x_nulls):
            self.step(int(w), bool(wn), int(x), bool(xn))
        return self
