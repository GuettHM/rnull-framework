"""Algebraic laws of R_null, checked on every backend (see docs/theory.pdf)."""

import numpy as np
import pytest

N = True   # null
R = False  # real


def assert_rnull(backend, x, values, mask):
    """Compare mask exactly and live values numerically (nan-aware)."""
    mask = np.asarray(mask, dtype=bool)
    np.testing.assert_array_equal(backend.mask(x), mask)
    got = backend.values(x)[~mask]
    np.testing.assert_allclose(got, np.asarray(values, dtype=float)[~mask], rtol=1e-12, atol=1e-12)


def random_pair(backend, rng, shape=(64,), p_null=0.4):
    return backend.make(rng.normal(size=shape), rng.random(shape) < p_null)


# ---------------------------------------------------------------- identities
def test_null_is_two_sided_additive_identity(backend):
    x = backend.make([5.5, -3.2, 0.0])
    null = backend.make([9.0, 9.0, 9.0], [N, N, N])
    assert_rnull(backend, x + null, [5.5, -3.2, 0.0], [R, R, R])
    assert_rnull(backend, null + x, [5.5, -3.2, 0.0], [R, R, R])


def test_null_is_two_sided_multiplicative_identity(backend):
    x = backend.make([5.5, -3.2, 0.0])
    null = backend.make([9.0, 9.0, 9.0], [N, N, N])
    assert_rnull(backend, x * null, [5.5, -3.2, 0.0], [R, R, R])
    assert_rnull(backend, null * x, [5.5, -3.2, 0.0], [R, R, R])


def test_null_combined_with_null_is_null(backend):
    a = backend.make([1.0, 2.0], [N, N])
    b = backend.make([3.0, 4.0], [N, N])
    for result in (a + b, a - b, a * b, a / b):
        assert backend.mask(result).all()
        np.testing.assert_array_equal(backend.values(result), [0.0, 0.0])


def test_measured_zero_is_not_null(backend):
    zero = backend.make([0.0])
    null = backend.make([0.0], [N])
    assert_rnull(backend, zero + null, [0.0], [R])
    assert_rnull(backend, zero * null, [0.0], [R])  # zero still annihilates


def test_hidden_payload_is_canonicalised(backend):
    x = backend.make([5.0, 7.0], [R, N])
    np.testing.assert_array_equal(backend.values(x), [5.0, 0.0])


# ------------------------------------------------------ monoid laws (random)
def test_commutativity(backend, rng):
    a, b = random_pair(backend, rng), random_pair(backend, rng)
    for op in ("__add__", "__mul__"):
        ab, ba = getattr(a, op)(b), getattr(b, op)(a)
        assert_rnull(backend, ab, backend.values(ba), backend.mask(ba))


def test_associativity(backend, rng):
    a, b, c = (random_pair(backend, rng) for _ in range(3))
    left, right = (a + b) + c, a + (b + c)
    np.testing.assert_array_equal(backend.mask(left), backend.mask(right))
    np.testing.assert_allclose(backend.values(left), backend.values(right), atol=1e-12)
    left, right = (a * b) * c, a * (b * c)
    np.testing.assert_array_equal(backend.mask(left), backend.mask(right))
    np.testing.assert_allclose(backend.values(left), backend.values(right), atol=1e-12)


def test_masks_are_homomorphic(backend, rng):
    a, b = random_pair(backend, rng), random_pair(backend, rng)
    ma, mb = backend.mask(a), backend.mask(b)
    np.testing.assert_array_equal(backend.mask(a + b), ma & mb)
    np.testing.assert_array_equal(backend.mask(a * b), ma & mb)
    np.testing.assert_array_equal(backend.mask(a / b), ma | mb)
    np.testing.assert_array_equal(backend.mask(a.synaptic_mul(b)), ma | mb)
    np.testing.assert_array_equal(backend.mask(a.synaptic_skip(b)), mb)


# ------------------------------------------------------- distributivity defect
@pytest.mark.parametrize(
    "b_null, c_null, expected_multiple",
    [(R, R, 0.0), (R, N, 1.0), (N, R, 1.0), (N, N, 1.0)],
)
def test_distributivity_defect(backend, b_null, c_null, expected_multiple):
    """For real a: (a*b + a*c) - a*(b + c) = a if b or c is null, else 0."""
    a = backend.make([7.5, -2.0, 0.0])
    b = backend.make([1.5, 3.0, 4.0], [b_null] * 3)
    c = backend.make([-0.5, 2.0, 1.0], [c_null] * 3)
    defect = ((a * b) + (a * c)) - (a * (b + c))
    assert_rnull(backend, defect, expected_multiple * np.array([7.5, -2.0, 0.0]), [R, R, R])


def test_distributivity_holds_when_multiplier_is_null(backend):
    a = backend.make([0.0], [N])
    b = backend.make([3.0], [N])
    c = backend.make([4.0])
    lhs, rhs = a * (b + c), (a * b) + (a * c)
    assert_rnull(backend, lhs, backend.values(rhs), backend.mask(rhs))


# ------------------------------------------------------------- null detector
def test_chi_null_mask_and_polynomial_agree_on_finite_values(backend):
    x = backend.make([-15.0, 0.0, 42.0, 1e30, 99.0], [R, R, R, R, N])
    np.testing.assert_array_equal(backend.plain(x.chi_null()), [0, 0, 0, 0, 1])
    np.testing.assert_array_equal(
        backend.plain(x.chi_null(evaluate_polynomial=True)), [0, 0, 0, 0, 1]
    )


@pytest.mark.filterwarnings("ignore:invalid value")
def test_chi_null_polynomial_is_undefined_for_infinite_values(backend):
    x = backend.make([np.inf])
    assert backend.plain(x.chi_null())[0] == 0.0  # the mask stays correct
    assert np.isnan(backend.plain(x.chi_null(evaluate_polynomial=True))[0])


# ------------------------------------------------------ subtraction / division
def test_subtraction_rules(backend):
    x = backend.make([4.0])
    null = backend.make([0.0], [N])
    assert_rnull(backend, x - null, [4.0], [R])
    assert_rnull(backend, null - x, [-4.0], [R])
    assert_rnull(backend, x - x, [0.0], [R])  # zero, not null: no additive inverses
    assert backend.mask(-null).all()


@pytest.mark.filterwarnings("ignore:divide by zero", "ignore:invalid value")
def test_division_rules(backend):
    x = backend.make([6.0, 1.0, 0.0])
    y = backend.make([3.0, 0.0, 0.0])
    q = x / y
    np.testing.assert_array_equal(backend.mask(q), [R, R, R])
    values = backend.values(q)
    assert values[0] == 2.0
    assert np.isposinf(values[1])  # real / 0 follows IEEE-754
    assert np.isnan(values[2])

    null = backend.make([0.0, 0.0, 0.0], [N, N, N])
    assert backend.mask(x / null).all()
    assert backend.mask(null / x).all()
    assert backend.mask(null / y).all()  # null / 0 is null, not inf


# ------------------------------------------------------- synaptic couplings
def test_synaptic_mul_is_absorbing(backend):
    w = backend.make([2.0, 2.0, 0.0, 0.0], [R, R, N, N])
    x = backend.make([3.0, 0.0, 3.0, 0.0], [R, N, R, N])
    assert_rnull(backend, w.synaptic_mul(x), [6.0, 0, 0, 0], [R, N, N, N])


def test_synaptic_skip_passes_input_through_null_weight(backend):
    w = backend.make([2.0, 2.0, 0.0, 0.0], [R, R, N, N])
    x = backend.make([3.0, 0.0, 3.0, 0.0], [R, N, R, N])
    assert_rnull(backend, w.synaptic_skip(x), [6.0, 0, 3.0, 0], [R, N, R, N])


# ----------------------------------------------------- reductions / coercion
def test_reductions_skip_nulls(backend):
    x = backend.make([2.0, 0.0, 4.0], [R, N, R])
    assert_rnull(backend, x.sum(), 6.0, R)
    assert_rnull(backend, x.prod(), 8.0, R)
    empty = backend.make([1.0, 2.0], [N, N])
    assert backend.mask(empty.sum()).all()
    assert backend.mask(empty.prod()).all()


def test_scalar_coercion(backend):
    x = backend.make([5.0, 0.0], [R, N])
    assert_rnull(backend, x + 1.0, [6.0, 1.0], [R, R])
    assert_rnull(backend, 2.0 * x, [10.0, 2.0], [R, R])
    assert_rnull(backend, 1.0 - x, [-4.0, 1.0], [R, R])


def test_product_fusion_magnitude_depends_on_presence(backend):
    """Documented behaviour: dropping a factor changes the value silently."""
    cam, radar = backend.make([2.0]), backend.make([4.0])
    lidar_on, lidar_off = backend.make([3.0]), backend.make([3.0], [N])
    assert_rnull(backend, cam * lidar_on * radar, [24.0], [R])
    assert_rnull(backend, cam * lidar_off * radar, [8.0], [R])


# ------------------------------------------- structure theorems (theory note)
def _grid(backend):
    """All triples over a small set of representatives, as three aligned arrays."""
    import itertools

    reps = [None, 0.0, 1.0, -1.0, 2.0, -3.5, 0.25]
    triples = list(itertools.product(reps, repeat=3))

    def column(i):
        values = [0.0 if t[i] is None else t[i] for t in triples]
        return backend.make(values, [t[i] is None for t in triples])

    return column(0), column(1), column(2)


def _same(backend, left, right):
    np.testing.assert_array_equal(backend.mask(left), backend.mask(right))
    np.testing.assert_allclose(backend.values(left), backend.values(right), atol=1e-12)


def test_oplus_and_synaptic_mul_form_a_commutative_semiring(backend):
    """(R_null, +, synaptic_mul) is a semiring with zero = null and one = 1."""
    a, b, c = _grid(backend)
    mul = lambda u, v: u.synaptic_mul(v)  # noqa: E731
    _same(backend, mul(a, b), mul(b, a))
    _same(backend, mul(mul(a, b), c), mul(a, mul(b, c)))
    _same(backend, mul(a, b + c), mul(a, b) + mul(a, c))      # distributivity
    one = backend.make(np.ones(len(backend.mask(a))))
    _same(backend, mul(one, a), a)
    null = backend.make(np.zeros(len(backend.mask(a))), np.ones(len(backend.mask(a)), dtype=bool))
    assert backend.mask(mul(null, a)).all()                    # zero is absorbing


def test_synaptic_skip_is_a_monoid_action(backend):
    """w |> x is an action of (R_null, *) on (R_null, +) by endomorphisms."""
    a, b, c = _grid(backend)
    _same(backend, (a * b).synaptic_skip(c), a.synaptic_skip(b.synaptic_skip(c)))
    _same(backend, a.synaptic_skip(b + c), a.synaptic_skip(b) + a.synaptic_skip(c))
    null = backend.make(np.zeros(len(backend.mask(a))), np.ones(len(backend.mask(a)), dtype=bool))
    _same(backend, null.synaptic_skip(a), a)                   # identity acts trivially
    assert backend.mask(a.synaptic_skip(null)).all()           # null is a fixed point


def test_distributivity_of_otimes_holds_exactly_when_predicted(backend):
    """a*(b+c) == a*b + a*c  iff  a is null, a == 0, or b and c are both real."""
    a, b, c = _grid(backend)
    lhs, rhs = a * (b + c), (a * b) + (a * c)
    holds = (backend.mask(lhs) == backend.mask(rhs)) & np.isclose(
        backend.values(lhs), backend.values(rhs), atol=1e-12
    )
    ma, mb, mc = backend.mask(a), backend.mask(b), backend.mask(c)
    predicted = ma | (~ma & (backend.values(a) == 0.0)) | (~mb & ~mc)
    np.testing.assert_array_equal(holds, predicted)
