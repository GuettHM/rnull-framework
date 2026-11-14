"""Shared fixtures: the algebra tests run against every available backend."""

import numpy as np
import pytest

from rnull import RNullArray


class NumpyBackend:
    name = "numpy"

    @staticmethod
    def make(values, is_null=None):
        return RNullArray(np.asarray(values, dtype=np.float64), is_null)

    @staticmethod
    def values(x):
        return np.asarray(x.v)

    @staticmethod
    def mask(x):
        return np.asarray(x.m)

    @staticmethod
    def plain(t):
        return np.asarray(t)


class TorchBackend:
    name = "torch"

    @staticmethod
    def make(values, is_null=None):
        import torch
        from rnull.core import RNullTensor

        return RNullTensor(torch.tensor(values, dtype=torch.float64), is_null=is_null)

    @staticmethod
    def values(x):
        return x.v.detach().cpu().numpy()

    @staticmethod
    def mask(x):
        return x.m.detach().cpu().numpy()

    @staticmethod
    def plain(t):
        return t.detach().cpu().numpy()


@pytest.fixture(params=["numpy", "torch"])
def backend(request):
    if request.param == "torch":
        pytest.importorskip("torch")
        return TorchBackend
    return NumpyBackend


@pytest.fixture
def rng():
    return np.random.default_rng(20261114)
