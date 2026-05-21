import numpy as np
from .module import Module
from ..tensor import Tensor
from ..ops import CrossEntropy

# The implementation of CrossEntropyLoss is provided for reference.
class CrossEntropyLoss(Module):
    def forward(self, pred: Tensor, target) -> Tensor:
        target_t = target if isinstance(target, Tensor) else Tensor(np.asarray(target), requires_grad=False)
        return CrossEntropy.apply(pred, target_t)
    
    
"""
Example usage:
    >>> loss_fn = MSELoss()  // Create an instance of the MSELoss module
    >>> pred = ...  //A tensor
    >>> target = ...    //Another tensor/scalar
    //This will call the forward method of MSELoss, and compute the MSE loss(a scalar tensor) using tensor operations(like +,-,mean(),...).
    >>> loss = loss_fn(pred, target)
    >>> loss.backward()
"""
class MSELoss(Module):
    def forward(self, pred: Tensor, target) -> Tensor:
        """
        Compute Mean Squared Error (MSE) loss.

        Inputs:
            pred (Tensor):
                Model predictions. May require gradients.
            target (Tensor or array-like):
                Ground-truth values. If `target` is not a Tensor, it is
                converted to a Tensor treated as a constant
                (i.e., `requires_grad=False`).

        Returns:
            Tensor:
                A scalar Tensor containing the mean squared error.
                
        Note:
            - Does not modify `pred` or `target` in-place.
        """
        if not isinstance(target, Tensor):
            target = Tensor(np.asarray(target, dtype=np.float32), requires_grad=False)
        diff = pred - target
        return (diff * diff).mean()
