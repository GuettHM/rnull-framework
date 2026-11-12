"""
Drift-Free Continual Learning Optimizers over R_null.
Author: Hakim Guettal
"""

import numpy as np
from .tensor import RNullTensor


class NullSGD:
    """
    R_null SGD Optimizer with Momentum and Weight Decay.
    Guarantees exact 0.000000 parameter drift on consolidated synapses (where grad = null).
    """
    def __init__(self, layers: list, lr: float = 0.01, momentum: float = 0.9, weight_decay: float = 1e-4):
        self.layers = layers
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [RNullTensor.full_null(layer.W.shape) for layer in layers]

    def step(self, grads_W: list):
        for i, (layer, grad_W) in enumerate(zip(self.layers, grads_W)):
            wd_term = RNullTensor(self.weight_decay * layer.W.v, grad_W.m.copy())
            eff_grad = grad_W.oplus(wd_term)

            vel = self.velocities[i]
            new_vel_v = np.where(
                eff_grad.m,
                vel.v,
                self.momentum * np.where(vel.m, 0.0, vel.v) + eff_grad.v
            )
            new_vel_m = vel.m & eff_grad.m
            self.velocities[i] = RNullTensor(new_vel_v, new_vel_m)

            delta_W = RNullTensor(-self.lr * self.velocities[i].v, eff_grad.m.copy())
            layer.W = layer.W.oplus(delta_W)


class NullAdam:
    """
    R_null Adam Optimizer with Decoupled Weight Decay Protection.
    Freezes first/second moment buffers and weight decay whenever grad_W = null.
    """
    def __init__(self, layers: list, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 1e-4):
        self.layers = layers
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m_buf = [np.zeros_like(layer.W.v) for layer in layers]
        self.v_buf = [np.zeros_like(layer.W.v) for layer in layers]
        self.t = 0

    def step(self, grads_W: list):
        self.t += 1
        for i, (layer, grad_W) in enumerate(zip(self.layers, grads_W)):
            active = ~grad_W.m
            g = grad_W.v

            self.m_buf[i] = np.where(active, self.beta1 * self.m_buf[i] + (1.0 - self.beta1) * g, self.m_buf[i])
            self.v_buf[i] = np.where(active, self.beta2 * self.v_buf[i] + (1.0 - self.beta2) * (g ** 2), self.v_buf[i])

            m_hat = self.m_buf[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v_buf[i] / (1.0 - self.beta2 ** self.t)

            step_val = -self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * layer.W.v)
            delta_W = RNullTensor(step_val, grad_W.m.copy())
            layer.W = layer.W.oplus(delta_W)
