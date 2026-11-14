"""PyTorch backend. Skipped entirely when PyTorch is not installed."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import rnull  # noqa: E402
from rnull import NullAdamW, NullLinear, RNullArray  # noqa: E402
from rnull.core import (  # noqa: E402
    RNullLinear,
    RNullTensor,
    continual_learning_step,
    protected_step,
)


def to_np(t):
    return t.detach().cpu().numpy()


def test_lazy_top_level_exports():
    assert rnull.RNullTensor is RNullTensor
    assert rnull.RNullLinear is RNullLinear


def test_dtype_handling():
    assert RNullTensor(torch.zeros(2, dtype=torch.float64)).v.dtype == torch.float64
    assert RNullTensor([1, 2, 3]).v.dtype == torch.get_default_dtype()


def test_linear_all_null_input_abstains_and_suppresses_bias():
    layer = RNullLinear(4, 2)
    layer.bias.data.copy_(torch.tensor([5.0, 5.0]))
    out = layer(RNullTensor.null_like(torch.zeros(3, 4)))
    assert to_np(out.m).all()
    assert np.all(to_np(out.v) == 0.0)


def test_linear_without_normalisation_equals_zero_imputation():
    torch.manual_seed(0)
    layer = RNullLinear(12, 5)
    values = torch.randn(32, 12)
    mask = torch.rand(32, 12) < 0.5
    mask[:, 0] = False
    out = layer(RNullTensor(values, is_null=mask))
    expected = np.where(to_np(mask), 0.0, to_np(values)) @ to_np(layer.weight).T + to_np(layer.bias)
    assert not to_np(out.m).any()
    np.testing.assert_allclose(to_np(out.v), expected, atol=1e-5)


def test_linear_normalisation_and_coverage():
    layer = RNullLinear(8, 1, bias=False, normalize=True)
    layer.weight.data.copy_(torch.ones(1, 8))
    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[0, :6] = True
    out, coverage = layer(RNullTensor(torch.ones(1, 8), is_null=mask), return_coverage=True)
    np.testing.assert_allclose(to_np(out.v), [[8.0]])
    np.testing.assert_allclose(to_np(coverage), [[0.25]])
    layer.normalize = "variance"
    np.testing.assert_allclose(to_np(layer(RNullTensor(torch.ones(1, 8), is_null=mask)).v), [[4.0]])
    with pytest.raises(ValueError):
        RNullLinear(2, 2, normalize="median")


def test_weight_null_mask():
    layer = RNullLinear(3, 2, bias=False)
    layer.weight.data.copy_(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    layer.set_weight_null_mask([[False, True, False], [True, True, True]])
    out = layer(RNullTensor(torch.ones(1, 3)))
    np.testing.assert_allclose(to_np(out.v), [[4.0, 0.0]])
    np.testing.assert_array_equal(to_np(out.m), [[False, True]])
    with pytest.raises(ValueError):
        layer.set_weight_null_mask([[True]])


def test_matches_numpy_reference_layer():
    rng = np.random.default_rng(5)
    W, b = rng.normal(size=(4, 9)), rng.normal(size=4)
    values, mask = rng.normal(size=(6, 9)), rng.random((6, 9)) < 0.4
    for normalize in (False, "mean", "variance"):
        ref = NullLinear(9, 4, normalize=normalize)
        ref.W, ref.b = RNullArray(W), RNullArray(b)
        expected = ref(RNullArray(values, mask))

        layer = RNullLinear(9, 4, normalize=normalize)
        layer.weight.data.copy_(torch.tensor(W, dtype=torch.float32))
        layer.bias.data.copy_(torch.tensor(b, dtype=torch.float32))
        got = layer(RNullTensor(torch.tensor(values, dtype=torch.float32), is_null=torch.tensor(mask)))
        np.testing.assert_array_equal(to_np(got.m), expected.m)
        np.testing.assert_allclose(to_np(got.v), expected.v, rtol=1e-4, atol=1e-5)


def test_continual_learning_step_is_a_masked_update():
    w = torch.nn.Parameter(torch.tensor([1.5, -2.0, 0.8, 4.2]))
    before = to_np(w).copy()
    frozen = torch.tensor([True, True, True, False])
    continual_learning_step(w, torch.tensor([-0.015, 0.020, -0.008, 0.250]), frozen)
    after = to_np(w)
    assert np.array_equal(after[:3], before[:3])
    np.testing.assert_allclose(after[3], 4.45, rtol=1e-6)


def test_protected_step_restores_frozen_entries():
    class HalvingOptimizer:
        def __init__(self, param):
            self.param = param

        def step(self):
            with torch.no_grad():
                self.param.copy_(self.param * 0.5)

    w = torch.nn.Parameter(torch.tensor([2.0, 4.0, 6.0]))
    protected_step(HalvingOptimizer(w), [(w, torch.tensor([True, False, True]))])
    np.testing.assert_array_equal(to_np(w), [2.0, 2.0, 6.0])


# ---------------------------------------------------------------------------
# The tests below need real autograd / torch.optim.
# ---------------------------------------------------------------------------
@pytest.mark.autograd
def test_gradients_flow_only_through_live_inputs():
    layer = RNullLinear(3, 1, bias=False)
    values = torch.ones(1, 3, requires_grad=True)
    mask = torch.tensor([[False, True, False]])
    layer(RNullTensor(values, is_null=mask)).v.sum().backward()
    grad = to_np(values.grad)[0]
    assert grad[1] == 0.0
    np.testing.assert_allclose(grad[[0, 2]], to_np(layer.weight)[0, [0, 2]])


@pytest.mark.autograd
def test_protected_step_with_adamw_weight_decay():
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(50))
    before = to_np(w).copy()
    frozen = torch.rand(50) < 0.5
    optimizer = torch.optim.AdamW([w], lr=1e-2, weight_decay=0.1)
    for _ in range(20):
        optimizer.zero_grad()
        (w ** 2).sum().backward()
        protected_step(optimizer, [(w, frozen)])
    after, frozen_np = to_np(w), to_np(frozen)
    assert np.array_equal(after[frozen_np], before[frozen_np])
    assert np.all(after[~frozen_np] != before[~frozen_np])


@pytest.mark.autograd
def test_numpy_adamw_matches_torch_adamw():
    class Box:
        pass

    rng = np.random.default_rng(1)
    start = rng.normal(size=20)
    box = Box()
    box.W = RNullArray(start)
    reference = NullAdamW([(box, "W")], lr=1e-2, weight_decay=0.05)

    w = torch.nn.Parameter(torch.tensor(start, dtype=torch.float64))
    optimizer = torch.optim.AdamW([w], lr=1e-2, weight_decay=0.05)
    for _ in range(30):
        grad = rng.normal(size=20)
        reference.step([RNullArray(grad)])
        optimizer.zero_grad()
        w.grad = torch.tensor(grad, dtype=torch.float64)
        optimizer.step()
    np.testing.assert_allclose(box.W.v, to_np(w), rtol=1e-8, atol=1e-10)
