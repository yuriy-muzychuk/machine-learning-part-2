"""
repr_learning_student.py — Representation Learning with Neural Networks (Student)
==================================================================================
Week 6 Lab: Learned vs engineered features; visualization and interpretation
of neural network hidden states.

Instructions
------------
Implement every method marked with
    raise NotImplementedError("TODO: ...")

Do NOT import anything beyond what is already imported.
Do NOT change method signatures, the __init__ constructor, or the forward method.

Tasks summary
-------------
1. RepresentationNet.encode(x)
       Forward pass through the hidden layers; return the bottleneck tensor.
2. train_epoch(model, optimizer, X_t, y_t)
       One complete gradient-descent step; return the scalar loss.
3. extract_activations(model, X_tensor, layer)
       Register a forward hook, run the model, capture and return activations.
4. linear_probe(h_train, y_train, h_val, y_val)
       Fit logistic regression on representations and return train/val accuracy.

Testing
-------
Run the notebook cells in order. Each Part exercises one or more of your
implementations. If a cell raises NotImplementedError, that is the function
you need to finish first.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Network Architecture
# ─────────────────────────────────────────────────────────────────────────────

class RepresentationNet(nn.Module):
    """Two-hidden-layer network with a 2-D bottleneck for representation learning.

    Architecture
    ------------
    Input(input_dim)
        → Linear(input_dim → hidden_dim) → ReLU           [act1]
        → Linear(hidden_dim → bottleneck_dim) → ReLU      [act2]
        → Linear(bottleneck_dim → n_classes)               [logits]

    The bottleneck layer (act2) intentionally keeps its output dimension at 2
    so that the learned representation can be plotted directly as a scatter plot,
    making the network's internal geometry visible without any dimensionality
    reduction.

    Parameters
    ----------
    input_dim      : int — number of input features (default 2)
    hidden_dim     : int — width of the first hidden layer (default 32)
    bottleneck_dim : int — width of the representation layer (default 2)
    n_classes      : int — number of output classes (default 2)
    """

    def __init__(self, input_dim=2, hidden_dim=32, bottleneck_dim=2, n_classes=2):
        super().__init__()
        self.fc1        = nn.Linear(input_dim,      hidden_dim)
        self.act1       = nn.ReLU()
        self.fc2        = nn.Linear(hidden_dim,     bottleneck_dim)
        self.act2       = nn.ReLU()
        self.classifier = nn.Linear(bottleneck_dim, n_classes)

    def encode(self, x):
        """Forward pass through the hidden layers only; return the bottleneck.

        Pass `x` through the following sequence of layers (in order):
            fc1  →  act1  →  fc2  →  act2
        and return the result. Do NOT apply the classifier head.

        Parameters
        ----------
        x : torch.Tensor shape (N, input_dim)

        Returns
        -------
        torch.Tensor shape (N, bottleneck_dim)

        Hint
        ----
        Each named sub-module is already defined in __init__ as an attribute
        (self.fc1, self.act1, self.fc2, self.act2). Call them in order:
            h = self.act1(self.fc1(x))
            z = self.act2(self.fc2(h))
            return z
        """
        raise NotImplementedError("TODO: implement encode")

    def forward(self, x):
        """Full forward pass: encode then classify.

        This method is already provided — do not modify it.
        It depends on your encode() implementation.
        """
        return self.classifier(self.encode(x))


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Training Utilities
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, optimizer, X_t, y_t):
    """Run one full training step using cross-entropy loss.

    Steps to implement (in order):
        1. Set the model to training mode:          model.train()
        2. Zero all parameter gradients:            optimizer.zero_grad()
        3. Forward pass to compute logits:          logits = model(X_t)
        4. Compute cross-entropy loss:              F.cross_entropy(logits, y_t)
        5. Backward pass:                           loss.backward()
        6. Optimizer step:                          optimizer.step()
        7. Return the scalar Python float:          loss.item()

    Parameters
    ----------
    model     : RepresentationNet (or any nn.Module)
    optimizer : torch.optim optimizer attached to model.parameters()
    X_t       : torch.Tensor shape (N, input_dim), dtype float32 — input features
    y_t       : torch.Tensor shape (N,), dtype long — integer class labels

    Returns
    -------
    float — scalar loss value for this epoch
    """
    raise NotImplementedError("TODO: implement train_epoch")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Activation Extraction via Hooks
# ─────────────────────────────────────────────────────────────────────────────

def extract_activations(model, X_tensor, layer):
    """Capture the output of a given layer using a PyTorch forward hook.

    A forward hook is a callback function that PyTorch calls automatically
    every time a forward pass runs through the module it is attached to.
    Use this mechanism to intercept the output of `layer` without modifying
    the model's structure or parameters.

    Steps to implement:
        1. Create a dict `captured = {}` to store the hook's result.
        2. Define a nested function `hook_fn(module, input, output)` that
           stores `output.detach()` in `captured['output']`.
        3. Register the hook on `layer`:
               handle = layer.register_forward_hook(hook_fn)
        4. Set model to eval mode and run a forward pass inside
           `with torch.no_grad():` so no gradients are computed.
        5. Remove the hook to avoid side effects:
               handle.remove()
        6. Return `captured['output'].numpy()`.

    Parameters
    ----------
    model    : nn.Module — the trained network
    X_tensor : torch.Tensor shape (N, input_dim) — input data
    layer    : nn.Module — a submodule of `model`, e.g. model.act1 or model.act2

    Returns
    -------
    np.ndarray shape (N, layer_output_dim)
    """
    raise NotImplementedError("TODO: implement extract_activations")


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Linear Probe
# ─────────────────────────────────────────────────────────────────────────────

def linear_probe(h_train, y_train, h_val, y_val):
    """Evaluate how linearly separable a representation is.

    Fit a logistic regression on the training representations and measure
    accuracy on both splits. A high validation accuracy means the network has
    organised the data into a form that a simple linear boundary can separate.

    Steps to implement:
        1. Create clf = LogisticRegression(max_iter=1000, random_state=42)
        2. Fit clf on h_train and y_train.
        3. Compute train accuracy:  accuracy_score(y_train, clf.predict(h_train))
        4. Compute val   accuracy:  accuracy_score(y_val,   clf.predict(h_val))
        5. Return (train_acc, val_acc).

    Parameters
    ----------
    h_train : np.ndarray shape (N_train, d) — training representations
    y_train : np.ndarray shape (N_train,)   — training labels
    h_val   : np.ndarray shape (N_val, d)   — validation representations
    y_val   : np.ndarray shape (N_val,)     — validation labels

    Returns
    -------
    train_acc : float
    val_acc   : float
    """
    raise NotImplementedError("TODO: implement linear_probe")
