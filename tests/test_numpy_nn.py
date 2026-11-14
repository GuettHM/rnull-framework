"""NumPy reference layers and optimisers."""

import numpy as np
import pytest

from rnull import (
    NullAdamW,
    NullLinear,
    NullMLP,
    NullSGD,
    RNullArray,
    fuse_product,
    present_count,
)


def test_repr_shows_nulls_for_any_rank():
    assert repr(RNullArray([5.0, 0.0, 7.0], [False, False, True])) == "RNullArray([5.0, 0.0, '∅'])"
    assert repr(RNullArray(5.0)) == "RNullArray(5.0)"
    assert repr(RNullArray(5.0, True)) == "RNullArray('∅')"
    assert repr(RNullArray(np.float32([[1.5]]))) == "RNullArray([[1.5]])"


def test_fuse_product_and_present_count():
    cam = RNullArray([2.0, 3.0])
    lidar = RNullArray([9.0, 9.0], [True, False])
    radar = RNullArray([4.0, 5.0])
    fused = fuse_product([cam, lidar, radar])
    np.testing.assert_allclose(fused.v, [8.0, 135.0])
    assert not fused.m.any()
    np.testing.assert_array_equal(present_count([cam, lidar, radar]), [2, 3])
    with pytest.raises(ValueError):
        fuse_product([])


def test_linear_all_null_input_abstains_and_suppresses_bias():
    layer = NullLinear(16, 8)
    layer.b = RNullArray(np.full(8, 5.0))
    out = layer(RNullArray.full_null((4, 16)))
    assert out.m.all()
    assert np.all(out.v == 0.0)
    np.testing.assert_array_equal(layer.last_coverage, 0.0)


def test_linear_without_normalisation_equals_zero_imputation(rng):
    layer = NullLinear(12, 5, seed=3)
    layer.b = RNullArray(rng.normal(size=5))
    values = rng.normal(size=(32, 12))
    mask = rng.random((32, 12)) < 0.5
    mask[:, 0] = False  # keep at least one live input per row
    out = layer(RNullArray(values, mask))
    expected = np.where(mask, 0.0, values) @ layer.W.v.T + layer.b.v
    assert not out.m.any()
    np.testing.assert_allclose(out.v, expected, atol=1e-12)


def test_linear_normalisation_rescales_by_active_fraction():
    layer = NullLinear(8, 1, normalize=True)
    layer.W = RNullArray(np.ones((1, 8)))
    mask = np.zeros((1, 8), dtype=bool)
    mask[0, :6] = True  # only 2 of 8 inputs are live
    out = layer(RNullArray(np.ones((1, 8)), mask))
    np.testing.assert_allclose(out.v, [[8.0]])  # 2 * (8 / 2)
    np.testing.assert_allclose(layer.last_coverage, [[0.25]])

    layer.normalize = "variance"
    np.testing.assert_allclose(layer(RNullArray(np.ones((1, 8)), mask)).v, [[4.0]])  # 2 * sqrt(8 / 2)

    layer.normalize = False
    np.testing.assert_allclose(layer(RNullArray(np.ones((1, 8)), mask)).v, [[2.0]])

    with pytest.raises(ValueError):
        NullLinear(2, 2, normalize="median")


def test_normalisation_modes_preserve_the_advertised_moment(rng):
    """For independent zero-mean inputs missing completely at random:
    'mean' is unbiased for the full sum, 'variance' keeps its second moment."""
    n_in, batch, missing = 64, 40_000, 0.6
    values = rng.standard_normal((batch, n_in))
    mask = rng.random((batch, n_in)) < missing
    weights = rng.standard_normal((1, n_in))

    def run(normalize, x):
        layer = NullLinear(n_in, 1, bias=False, normalize=normalize)
        layer.W = RNullArray(weights)
        out = layer(x)
        return out.v[~out.m]

    full = run(False, RNullArray(values))
    mean_mode = run("mean", RNullArray(values, mask))
    var_mode = run("variance", RNullArray(values, mask))
    plain = run(False, RNullArray(values, mask))

    # 'variance' restores the scale; plain shrinks it; 'mean' overshoots it.
    assert abs(var_mode.std() / full.std() - 1.0) < 0.05
    assert plain.std() / full.std() < 0.7
    assert mean_mode.std() / full.std() > 1.4
    # 'mean' is conditionally unbiased: regressing it on the full sum gives slope 1.
    slope = float(np.dot(mean_mode, full) / np.dot(full, full))
    assert abs(slope - 1.0) < 0.05


def test_linear_null_weights_are_skipped():
    layer = NullLinear(3, 2, bias=False)
    layer.W = RNullArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[False, True, False], [True, True, True]])
    out = layer(RNullArray([[1.0, 1.0, 1.0]]))
    np.testing.assert_allclose(out.v, [[4.0, 0.0]])
    np.testing.assert_array_equal(out.m, [[False, True]])  # neuron 1 has no live synapse
    assert layer.last_metrics["executed_macs"] == 2
    assert layer.last_metrics["skipped_macs"] == 4


def test_linear_rejects_wrong_shape():
    with pytest.raises(ValueError):
        NullLinear(4, 2)(RNullArray(np.zeros((3, 5))))


@pytest.mark.parametrize("normalize", [False, "mean", "variance"])
def test_backward_matches_finite_differences(rng, normalize):
    mlp = NullMLP([6, 5, 3], alpha=0.1, seed=11, normalize=normalize)
    w_mask = rng.random(mlp.layers[0].W.shape) < 0.2
    mlp.layers[0].W = RNullArray(mlp.layers[0].W.v, w_mask)
    x_mask = rng.random((7, 6)) < 0.3
    x_mask[:, 0] = False
    X = RNullArray(rng.normal(size=(7, 6)), x_mask)

    def loss() -> float:
        out = mlp.forward(X)
        return 0.5 * float(np.sum(out.v ** 2))

    out = mlp.forward(X)
    grads = mlp.backward(RNullArray(out.v, out.m))
    eps = 1e-6
    for (owner, name), grad in zip(mlp.parameters(), grads):
        param = getattr(owner, name)
        numeric = np.zeros(param.shape)
        for idx in np.ndindex(param.shape):
            if param.m[idx]:
                continue
            base = param.v[idx]
            param.v[idx] = base + eps
            up = loss()
            param.v[idx] = base - eps
            down = loss()
            param.v[idx] = base
            numeric[idx] = (up - down) / (2 * eps)
        live = ~grad.m
        np.testing.assert_allclose(grad.v[live], numeric[live], rtol=1e-5, atol=1e-7)
        # wherever the analytic gradient is null, the true gradient is zero
        np.testing.assert_allclose(numeric[grad.m], 0.0, atol=1e-7)
        if name == "W":
            assert grad.m[param.m].all()  # null weights never receive a gradient


@pytest.mark.parametrize("optimizer_cls", [NullSGD, NullAdamW])
def test_consolidated_weights_do_not_drift(rng, optimizer_cls):
    mlp = NullMLP([32, 64, 16], seed=123)
    reference = mlp.layers[0].W.v.copy()
    frozen = rng.random(reference.shape) < 0.5
    mlp.layers[0].consolidated_mask = frozen

    optimizer = optimizer_cls(mlp.parameters(), lr=1e-2, weight_decay=1e-2)
    for _ in range(25):
        X = RNullArray(rng.normal(size=(16, 32)), rng.random((16, 32)) < 0.3)
        mlp.forward(X)
        optimizer.step(mlp.backward(RNullArray(rng.normal(size=(16, 16)))))

    updated = mlp.layers[0].W.v
    assert np.array_equal(updated[frozen], reference[frozen])      # bit-exact
    assert np.all(updated[~frozen] != reference[~frozen])          # the rest did move


def test_optimizer_keeps_null_parameters_null(rng):
    layer = NullLinear(4, 3, seed=1)
    pruned = np.zeros((3, 4), dtype=bool)
    pruned[0, :2] = True
    layer.W = RNullArray(layer.W.v, pruned)
    optimizer = NullAdamW(layer.parameters(), lr=1e-2)
    for _ in range(5):
        layer.forward(RNullArray(rng.normal(size=(8, 4))))
        _, grad_W, grad_b = layer.backward(RNullArray(rng.normal(size=(8, 3))))
        optimizer.step([grad_W, grad_b])
    np.testing.assert_array_equal(layer.W.m, pruned)


def test_optimizer_validates_gradient_count():
    layer = NullLinear(2, 2)
    with pytest.raises(ValueError):
        NullSGD(layer.parameters()).step([])
    with pytest.raises(ValueError):
        NullSGD([])
