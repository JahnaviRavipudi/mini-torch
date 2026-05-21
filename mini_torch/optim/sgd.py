"""
Example usage:
    >>> optimizer = SGD(model.parameters(), lr=0.01)  // Create an instance of the SGD optimizer with model parameters and learning rate 0.01
    >>> optimizer.zero_grad()  // Clear gradients of all model parameters
    >>> loss.backward()  // Compute gradients of the loss with respect to model parameters
    >>> optimizer.step()  // Update model parameters using the computed gradients and the learning rate
"""
class SGD:
    """
    Stochastic Gradient Descent (SGD) optimizer.

    SGD performs a simple first-order update on a set of learnable parameters.
    For each parameter `p` with gradient `p.grad`, the update is:
        p <- p - lr * p.grad
    """

    def __init__(self, params, lr=1e-2):
        self.params = list(params)
        self.lr = float(lr)

    def step(self):
        """
        Apply one SGD update to all managed parameters.

        Inputs:
            None

        Returns:
            None

        Side Effects:
            - Modifies parameter values in-place (updates `p.data`).
            - Does not modify `p.grad`.
        """
        for p in self.params:
            if p.grad is None:
                continue
            p.data = p.data - self.lr * p.grad

    def zero_grad(self):
        """
        Clear accumulated gradients for all managed parameters.

        Inputs:
            None

        Returns:
            None

        Side Effects:
            - Resets each parameter's stored gradient (e.g., sets `p.grad = None`)
              so subsequent backward passes start fresh and do not accumulate.
        """
        for p in self.params:
            p.grad = None
