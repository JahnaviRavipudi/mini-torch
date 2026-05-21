import numpy as np
from typing import Any, Optional, Set, List, TYPE_CHECKING


def as_tensor(x: Any) -> "Tensor":
    return x if isinstance(x, Tensor) else Tensor(x, requires_grad=False)

class Tensor:
    """
    Tensor object representing data in the computation graph.

    A Tensor stores numerical data and gradient-related information. When
    requires_grad is True, it participates in automatic differentiation and
    keeps a reference to the Function (grad_fn) that created it, enabling
    gradient propagation during backward().
    """
    def __init__(self, data: Any, requires_grad: bool = False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = bool(requires_grad)
        self.grad: Optional[np.ndarray] = None
        from .ops import Function
        self.grad_fn: Optional[Function] = None  

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = None

    def numpy(self) -> np.ndarray:
        if self.requires_grad:
            raise RuntimeError(
                "Can't call numpy() on Tensor that requires grad. "
                "Use tensor.detach().numpy() instead."
            )
        return self.data
    
    def detach(self) -> "Tensor":
        """
        Return a new Tensor sharing the same data but without gradient tracking.
        """
        out = Tensor(self.data, requires_grad=False)
        return out
    
    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

    # ---- ops ----
    def __add__(self, other):  # Tensor + any
        """    
        When you write:   a + b
        Python actually calls:  a.__add__(b)
        So implementing __add__ lets our Tensor support the `+` operator.
        """
        from .ops import Add
        return Add.apply(self, other)

    def __radd__(self, other):  # any + Tensor
        from .ops import Add
        return Add.apply(other, self)

    def __mul__(self, other):
        from .ops import Mul
        return Mul.apply(self, other)

    def __rmul__(self, other):
        from .ops import Mul
        return Mul.apply(other, self)

    def __neg__(self):
        from .ops import Neg
        return Neg.apply(self)

    def __sub__(self, other):
        from .ops import Add, Neg
        return Add.apply(self, Neg.apply(other))

    def __rsub__(self, other):
        from .ops import Add, Neg
        return Add.apply(other, Neg.apply(self))

    def __matmul__(self, other):
        from .ops import MatMul
        return MatMul.apply(self, other)

    def __pow__(self, power):
        from .ops import Pow
        return Pow.apply(self, power)

    def __truediv__(self, other):
        return self * (as_tensor(other) ** -1)

    def __rtruediv__(self, other):
        return as_tensor(other) * (self ** -1)

    def sum(self):
        from .ops import Sum
        return Sum.apply(self)

    def mean(self):
        from .ops import Mean
        return Mean.apply(self)

    def relu(self):
        from .ops import ReLU
        return ReLU.apply(self)

    def sigmoid(self):
        from .ops import Sigmoid
        return Sigmoid.apply(self)

    def backward(self):
        """
        Backpropagate gradients from this scalar Tensor.

        Side Effects:
        - Traverses the computation graph reachable from `self` in reverse topological order.
        - For each `Function` node, calls `backward(ctx, grad_out)` to produce gradients for its parent tensors.
        - Accumulates gradients into `.grad` **only for leaf tensors** with `requires_grad=True` (non-leaf tensors do not keep persistent `.grad` in `mini_torch`).
        - By default, frees the computation graph after the backward pass by breaking references held by non-leaf nodes:
            - Clear each non-leaf tensor's `grad_fn` pointer.
            - Clear the `Function` node's `parents` list.
            - Clear any saved objects in `ctx` (e.g., `saved_tensors`) and drop `ctx` itself.
        TODO:
        1. Collect all reachable tensors in topological order via `grad_fn.parents`.
        2. Initialize the upstream gradient for `self` as `1`.
        3. Traverse in reverse order: propagate gradients through `grad_fn.backward(...)` and accumulate into parents (leaves into `.grad`).
        """

        if not self.requires_grad:
            return

        if self.data.size != 1:
            raise ValueError("backward() can only be called on a scalar Tensor.")

        # 1. Topological sort: collect all tensors reachable from self
        topo_order = []
        visited = set()

        def _topo(t):
            if id(t) in visited:
                return
            visited.add(id(t))
            if t.grad_fn is not None:
                for parent in t.grad_fn.parents:
                    _topo(parent)
            topo_order.append(t)

        _topo(self)

        # 2. Initialize gradient for self
        grad_map = {id(self): np.ones_like(self.data, dtype=np.float32)}

        # 3. Traverse in reverse topological order (from output to inputs)
        for t in reversed(topo_order):
            grad_out = grad_map.get(id(t), None)
            if grad_out is None:
                continue
            if t.grad_fn is None:
                # Leaf node
                continue
            # Compute gradients for parents
            grads = t.grad_fn.backward(t.grad_fn.ctx, grad_out)
            for parent, g in zip(t.grad_fn.parents, grads):
                if g is None:
                    continue
                if not parent.requires_grad:
                    continue
                if id(parent) in grad_map:
                    grad_map[id(parent)] = grad_map[id(parent)] + g
                else:
                    grad_map[id(parent)] = g

        # 4. Accumulate gradients into leaf tensors
        for t in topo_order:
            if t.grad_fn is None and t.requires_grad:
                g = grad_map.get(id(t), None)
                if g is not None:
                    if t.grad is None:
                        t.grad = g.astype(np.float32)
                    else:
                        t.grad = (t.grad + g).astype(np.float32)

        # 5. Free the computation graph
        for t in topo_order:
            if t.grad_fn is not None:
                t.grad_fn.ctx.saved_tensors = ()
                t.grad_fn.ctx.saved_values = ()
                t.grad_fn.ctx = None
                t.grad_fn.parents = ()
                t.grad_fn = None

