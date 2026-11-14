"""R_null: masked real arithmetic with a null element that acts as the
identity of both addition and multiplication.

The NumPy reference implementation is always available::

    from rnull import RNullArray, NullLinear, NullMLP, NullAdamW

The PyTorch implementation needs the optional ``torch`` extra and is loaded
on first use::

    from rnull import RNullTensor, RNullLinear          # or: from rnull.core import ...
"""

from .hw_model import RNullMacModel
from .nn import MultimodalProductFusion, NullLinear, NullMLP, fuse_product, present_count
from .optim import NullAdam, NullAdamW, NullSGD
from .tensor import RNullArray

__version__ = "1.1.0"

_TORCH_EXPORTS = (
    "RNullTensor",
    "RNullLinear",
    "continual_learning_step",
    "protected_step",
)

__all__ = [
    "RNullArray",
    "fuse_product",
    "present_count",
    "MultimodalProductFusion",
    "NullLinear",
    "NullMLP",
    "NullSGD",
    "NullAdamW",
    "NullAdam",
    "RNullMacModel",
    *_TORCH_EXPORTS,
]


def __getattr__(name):
    if name in _TORCH_EXPORTS:
        try:
            from . import core
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                f"rnull.{name} requires PyTorch. Install it with "
                "'pip install \"rnull-framework[torch]\"'."
            ) from exc
        return getattr(core, name)
    raise AttributeError(f"module 'rnull' has no attribute {name!r}")
