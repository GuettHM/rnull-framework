"""
rnull: Causal Transparency Algebra (R_null) and Null-Net Neural Architecture
Author: Hakim Guettal
License: MIT
"""

from .tensor import RNullTensor, causal_coupling
from .nn import NullLinear, MultimodalProductFusion, NullMLP
from .optim import NullSGD, NullAdam

__version__ = "1.0.0"
__author__ = "Hakim Guettal"
__all__ = [
    "RNullTensor",
    "causal_coupling",
    "NullLinear",
    "MultimodalProductFusion",
    "NullMLP",
    "NullSGD",
    "NullAdam",
]
