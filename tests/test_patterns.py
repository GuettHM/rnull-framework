import torch.nn as nn
import torch
from rnull.core import RNullTensor, RNullLinear, continual_learning_step


def test_bi_identity_axioms_and_chi_null():
    x = RNullTensor([5.0, 0.0, -3.2], is_null=[False, False, True])
    null_t = RNullTensor.null_like(x)

    # Axiom 1: x (+) null = x and x (*) null = x
    add_res = x + null_t
    mul_res = x * null_t
    assert torch.allclose(add_res.v[:2], x.v[:2])
    assert torch.allclose(mul_res.v[:2], x.v[:2])
    assert torch.equal(add_res.m, x.m)
    assert torch.equal(mul_res.m, x.m)

    # Exact equivalence between boolean mask and algebraic polynomial chi_null(x)
    assert torch.equal(x.chi_null(evaluate_polynomial=False), x.chi_null(evaluate_polynomial=True))


def test_multimodal_sensor_fusion_and_ood():
    s_cam = RNullTensor([2.0, 4.0], is_null=[False, False])
    s_lidar = RNullTensor([999.0, -999.0], is_null=[True, True])  # Offline sensor = null
    s_radar = RNullTensor([3.0, 5.0], is_null=[False, False])

    fused = s_cam * s_lidar * s_radar
    assert torch.allclose(fused.v, torch.tensor([6.0, 20.0]))
    assert not fused.m.any()

    # Total sensor failure -> OOD native abstention (Y = null)
    all_dead = s_lidar * RNullTensor.null_like(s_cam)
    assert all_dead.m.all()


def test_continual_learning_zero_drift():
    w = torch.nn.Parameter(torch.tensor([1.5, -2.0, 0.8, 4.2]))
    w_initial = w.detach().clone()
    frozen_mask = torch.tensor([True, True, True, False])
    # Simulated AdamW update containing non-zero weight decay & momentum on frozen weights
    raw_delta = torch.tensor([-0.015, 0.020, -0.008, 0.250])

    continual_learning_step(w, raw_delta, frozen_mask)
    mse_frozen = torch.mean((w[:3] - w_initial[:3]) ** 2).item()
    assert mse_frozen == 0.0
    assert torch.isclose(w[3], torch.tensor(4.45))


def test_synaptic_mac_and_linear_ood_abstention():
    layer = RNullLinear(4, 2, bias=True)
    nn.init.constant_(layer.bias, 5.0)  # Non-zero bias to verify no hallucination on null input

    null_input = RNullTensor.null_like(torch.zeros(2, 4))
    out = layer(null_input)
    assert out.m.all()
    assert torch.all(out.v == 0.0)
