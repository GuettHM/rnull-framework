"""Bit-accurate MAC model versus the software algebra."""

import numpy as np
import pytest

from rnull import RNullArray, RNullMacModel


def test_reset_state_is_null():
    mac = RNullMacModel()
    assert mac.acc_null and mac.acc_val == 0 and not mac.acc_ovf
    assert mac.acc_width == 40


def test_null_cycles_hold_state():
    mac = RNullMacModel()
    mac.step(3, False, 4, False)
    mac.step(100, True, 100, False)
    mac.step(100, False, 100, True)
    mac.step(100, True, 100, True)
    assert (mac.acc_val, mac.acc_null) == (12, False)
    assert (mac.cycles, mac.active_cycles) == (4, 1)


def test_zero_product_is_real_not_null():
    mac = RNullMacModel()
    mac.step(0, False, 123, False)
    assert mac.acc_val == 0 and not mac.acc_null


def test_clear_returns_to_null():
    mac = RNullMacModel()
    mac.step(2, False, 2, False)
    mac.clear()
    assert mac.acc_null and mac.acc_val == 0


def test_matches_software_accumulation(rng):
    for _ in range(20):
        n = 200
        w = rng.integers(-2**15, 2**15, n)
        x = rng.integers(-2**15, 2**15, n)
        w_null = rng.random(n) < 0.5
        x_null = rng.random(n) < 0.5
        mac = RNullMacModel().run(w, w_null, x, x_null)

        products = RNullArray(w.astype(np.float64), w_null).synaptic_mul(
            RNullArray(x.astype(np.float64), x_null)
        )
        total = products.sum()
        assert mac.acc_null == bool(total.m)
        assert not mac.acc_ovf
        assert mac.acc_val == int(total.v)  # exact: |sum| < 2**53
        assert mac.active_cycles == int(products.count_present())


def test_all_null_stream_stays_null():
    mac = RNullMacModel().run([1, 2, 3], [True] * 3, [4, 5, 6], [False] * 3)
    assert mac.acc_null and mac.active_cycles == 0


def test_guard_bits_bound_overflow():
    # 2 * 4 + 1 = 9-bit accumulator: range [-256, 255]; (-8) * (-8) = 64.
    mac = RNullMacModel(width=4, guard=1)
    for _ in range(3):
        mac.step(-8, False, -8, False)
    assert mac.acc_val == 192 and not mac.acc_ovf
    mac.step(-8, False, -8, False)
    assert mac.acc_ovf and mac.acc_val == -256  # wrapped, and flagged

    # With the default 8 guard bits, 2**8 worst-case products cannot overflow.
    mac = RNullMacModel()
    for _ in range(256):
        mac.step(-2**15, False, -2**15, False)
    assert not mac.acc_ovf and mac.acc_val == 256 * 2**30


def test_operand_range_and_parameters_are_validated():
    with pytest.raises(ValueError):
        RNullMacModel().step(2**15, False, 0, False)
    with pytest.raises(ValueError):
        RNullMacModel(guard=0)
