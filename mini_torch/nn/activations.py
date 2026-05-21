from .module import Module
from ..tensor import Tensor

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        from ..ops import ReLU
        return ReLU.apply(x)

class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        from ..ops import Sigmoid
        return Sigmoid.apply(x)

