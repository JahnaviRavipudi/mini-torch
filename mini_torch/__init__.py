from .tensor import Tensor
from . import nn
from . import optim
from . import utils

def tensor(data, requires_grad=False):
    return Tensor(data, requires_grad=requires_grad)

__all__ = [
    "Tensor",
    "nn",
    "optim",
    "utils",
]