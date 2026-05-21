# mini_torch

A minimal PyTorch-style deep learning library built from scratch using only NumPy. Implements dynamic computation graphs, reverse-mode automatic differentiation, neural network layers, and an SGD optimizer. Trains a 3-layer MLP on MNIST and hits 95%+ test accuracy.

Built as part of graduate coursework in machine learning at UIUC.

## Overview

The goal was to understand what happens under the hood when you call `.backward()` in PyTorch. This library re-implements the core pieces: a `Tensor` class that tracks operations, a `Function` base class for defining differentiable ops, and a `backward()` method that walks the computation graph in reverse topological order to propagate gradients.

## Quick start

```python
import mini_torch as torch
import mini_torch.nn as nn
from mini_torch.optim import SGD

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

model = MLP()
optimizer = SGD(model.parameters(), lr=0.03)
loss_fn = nn.CrossEntropyLoss()

for xb, yb in train_loader:
    loss = loss_fn(model(xb), yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## How the autograd engine works

Each forward op (`Add`, `Mul`, `MatMul`, etc.) is a subclass of `Function`. When you call an op, `Function.apply()` does three things: runs the forward computation on raw NumPy arrays, saves whatever the backward pass will need into a `Context` object, and wires the output tensor's `grad_fn` back to the op node. This builds a DAG on the fly.

`Tensor.backward()` then does a DFS from the loss node to collect every reachable tensor, sorts them topologically, and walks backward — calling each node's `backward()` to push gradients to its parents. Leaf tensors (i.e., model parameters) accumulate gradients in `.grad`. Once the pass is done, the graph is freed to avoid memory leaks.

## What's implemented

**Tensor ops** — `Add`, `Mul`, `Neg`, `Pow`, `MatMul`, `Sum`, `Mean`, `ReLU`, `Sigmoid`, `CrossEntropy`. All support broadcasting; `MatMul` handles 1D inputs by reshaping internally. Gradients are un-broadcast via `_unbroadcast()` so shapes always match.

**nn module** — `Module` base class with recursive `parameters()` collection, `Linear` layer with Kaiming init, `ReLU`/`Sigmoid` activations, `MSELoss` and `CrossEntropyLoss`.

**Optimizer** — vanilla SGD.

**Data utilities** — `Dataset`, `TensorDataset`, and a `DataLoader` with batching and shuffling.

## Repo structure

```
mini_torch/
├── tensor.py              # Tensor class, backward algorithm
├── ops.py                 # Function base + all differentiable ops
├── nn/
│   ├── module.py          # Module base, parameter traversal
│   ├── layers.py          # Linear
│   ├── activations.py     # ReLU, Sigmoid
│   └── losses.py          # MSELoss, CrossEntropyLoss
├── optim/
│   └── sgd.py             # SGD
└── utils/data/
    ├── dataset.py         # Dataset, TensorDataset
    └── dataloader.py      # DataLoader

mnist_classification.py    # End-to-end MNIST training
```

## MNIST results

3-layer MLP (`784 → 256 → 128 → 10`), SGD lr=0.03, batch size 128, per-pixel normalization.

| Epoch | Test Loss | Test Acc |
|-------|-----------|----------|
| 1     | ~0.45     | ~88%     |
| 5     | ~0.18     | ~94%     |
| 10    | ~0.12     | >95%     |

## Running it

```bash
pip install numpy
# Place MNIST IDX files (gzipped) in data/mnist/, then:
python mnist_classification.py
```

Python 3.10+.

## Key design decisions

**Broadcasting-aware gradients** — when an input gets broadcast during the forward pass (e.g., adding a `(out_features,)` bias to a `(batch, out_features)` tensor), the backward pass needs to sum the gradient back down to the original shape. `_unbroadcast()` handles this so gradient shapes always match, regardless of how many dimensions were expanded.

**Graph cleanup after backward** — once `backward()` finishes, all `grad_fn`, `ctx`, and `parents` references are cleared. This prevents stale computation graphs from piling up in memory and avoids accidental double-backward calls.

**Leaf gradient accumulation** — model parameters accumulate gradients across `backward()` calls (`p.grad += new_grad`), matching PyTorch semantics. This is why `optimizer.zero_grad()` is needed before each backward pass.

**Kaiming initialization** — `Linear` layers use `U(-1/√fan_in, 1/√fan_in)` for weight init, which keeps activations stable through deep ReLU networks. Compared to plain random normal init, this made a noticeable difference in convergence speed.

**Numerically stable cross-entropy** — the `CrossEntropy` op shifts logits by `max(logits)` before computing softmax (log-sum-exp trick) to prevent float32 overflow. Supports both integer class labels and full probability distributions as targets.
