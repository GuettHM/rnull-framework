"""
Automated Verification Suite for Causal Transparency Algebra (R_null) and Null-Net.
Author: Hakim Guettal
"""

import unittest
import numpy as np
from rnull import RNullTensor, NullLinear, MultimodalProductFusion, NullMLP, NullSGD, NullAdam


class TestCausalTransparencyAlgebra(unittest.TestCase):

    def test_axiom1_bilateral_identity(self):
        x = RNullTensor([5.5, -3.2, 0.0])
        null = RNullTensor.full_null((3,))
        res_add1 = x + null
        res_add2 = null + x
        res_mul1 = x * null
        res_mul2 = null * x

        np.testing.assert_allclose(res_add1.v, x.v)
        np.testing.assert_allclose(res_add2.v, x.v)
        np.testing.assert_allclose(res_mul1.v, x.v)
        np.testing.assert_allclose(res_mul2.v, x.v)
        self.assertFalse(np.any(res_add1.m))
        self.assertFalse(np.any(res_mul1.m))

    def test_axiom2_quotient_collapse(self):
        x = RNullTensor([10.0, 0.0])
        null = RNullTensor.full_null((2,))
        q1 = x / null
        q2 = null / x
        self.assertTrue(np.all(q1.m))
        self.assertTrue(np.all(q2.m))

    def test_causal_arrow_theorem(self):
        a = RNullTensor([7.5])
        zero = RNullTensor([0.0])
        null = RNullTensor.full_null((1,))

        lhs = a * (zero + null)
        rhs = (a * zero) + (a * null)

        self.assertEqual(lhs.v[0], 0.0)
        self.assertEqual(rhs.v[0], 7.5)
        self.assertNotEqual(lhs.v[0], rhs.v[0])

    def test_causal_defect_detector_chi(self):
        x = RNullTensor([-15.0, 0.0, 42.0, 99.0], m=[False, False, False, True])
        chi = x.causal_defect_chi()
        np.testing.assert_allclose(chi.v, [0.0, 0.0, 0.0, 1.0])
        self.assertFalse(np.any(chi.m))

    def test_multimodal_fusion(self):
        cam = RNullTensor([2.0, 3.0])
        lidar_off = RNullTensor.full_null((2,))
        radar = RNullTensor([4.0, 5.0])
        fusion = MultimodalProductFusion()
        out = fusion.forward([cam, lidar_off, radar])
        np.testing.assert_allclose(out.v, [8.0, 15.0])
        self.assertFalse(np.any(out.m))

    def test_epistemic_abstention(self):
        layer = NullLinear(16, 8)
        x_vac = RNullTensor.full_null((16, 4))
        y = layer.forward(x_vac)
        self.assertTrue(np.all(y.m))

    def test_zero_catastrophic_forgetting(self):
        mlp = NullMLP([32, 64, 16], seed=123)
        rng = np.random.default_rng(123)
        w0_ref = mlp.layers[0].W.v.copy()
        mask0 = rng.random(w0_ref.shape) < 0.5
        mlp.layers[0].consolidated_mask = mask0

        opt = NullAdam(mlp.layers, lr=1e-2, weight_decay=1e-2)
        for _ in range(25):
            X = RNullTensor(rng.normal(0, 1, (32, 16)), rng.random((32, 16)) < 0.3)
            Y_pred = mlp.forward(X)
            grad_loss = RNullTensor(rng.normal(0, 1, (16, 16)))
            grads = mlp.backward(grad_loss)
            opt.step(grads)

        mse_drift = np.mean((mlp.layers[0].W.v[mask0] - w0_ref[mask0]) ** 2)
        self.assertEqual(mse_drift, 0.0)


if __name__ == "__main__":
    unittest.main()
