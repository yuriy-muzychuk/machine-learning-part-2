"""
nn_student.py - Two-Layer Neural Network (Student Version)
===========================================================
Week 3 Lab: Implement a two-layer neural network from scratch using NumPy.

Instructions
------------
This module contains the building blocks of a two-layer neural network:

    Input (n_features)
        → Hidden layer (n_hidden neurons, tanh activation)
        → Output layer (1 neuron, sigmoid activation)

Each function below has a docstring explaining what it should do,
the expected input/output shapes, and a hint. Replace every
    raise NotImplementedError("TODO: ...")
with your own implementation.

Do NOT import anything beyond what is already imported at the top.
Do NOT change function signatures.

Testing
-------
Run the notebook Part 1 cells to check your implementations step by step.
The gradient check in Part 2 will catch most bugs automatically.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────
# Section 1 - Activation Functions
# ─────────────────────────────────────────────────────────────

def sigmoid(z):
    """Sigmoid activation: σ(z) = 1 / (1 + exp(-z)).

    Parameters
    ----------
    z : np.ndarray, any shape

    Returns
    -------
    np.ndarray, same shape as z, values in (0, 1)

    Hint: use np.exp. Be careful about numerical overflow for large
    negative values - np.exp(-z) can explode. np.clip(z, -500, 500)
    before exponentiating is a safe guard.
    """
    raise NotImplementedError("TODO: implement sigmoid")


def sigmoid_derivative(z):
    """Derivative of sigmoid: σ'(z) = σ(z) · (1 − σ(z)).

    Parameters
    ----------
    z : np.ndarray, any shape  ← the PRE-ACTIVATION (not the output!)

    Returns
    -------
    np.ndarray, same shape as z

    Hint: call sigmoid(z) and use the formula above.
    """
    raise NotImplementedError("TODO: implement sigmoid_derivative")


def tanh_derivative(z):
    """Derivative of tanh: tanh'(z) = 1 − tanh²(z).

    Parameters
    ----------
    z : np.ndarray, any shape  ← the PRE-ACTIVATION

    Returns
    -------
    np.ndarray, same shape as z

    Hint: use np.tanh(z).
    """
    raise NotImplementedError("TODO: implement tanh_derivative")


# ─────────────────────────────────────────────────────────────
# Section 2 - Weight Initialisation
# ─────────────────────────────────────────────────────────────

def initialise_weights(n_input, n_hidden, n_output, seed=42):
    """Randomly initialise network parameters.

    Use Xavier (Glorot) initialisation for weights:
        W ~ N(0, 1) * sqrt(1 / n_in)
    where n_in is the number of inputs to that layer.
    Initialise all biases to zero.

    Parameters
    ----------
    n_input  : int - number of input features
    n_hidden : int - number of hidden neurons
    n_output : int - number of output neurons (1 for binary classification)
    seed     : int - random seed for reproducibility

    Returns
    -------
    params : dict with keys
        'W1' : np.ndarray shape (n_hidden, n_input)
        'b1' : np.ndarray shape (n_hidden,)
        'W2' : np.ndarray shape (n_output, n_hidden)
        'b2' : np.ndarray shape (n_output,)

    Hint: use np.random.default_rng(seed).normal(...)
    Scale W1 by sqrt(1/n_input) and W2 by sqrt(1/n_hidden).
    """
    raise NotImplementedError("TODO: implement initialise_weights")


# ─────────────────────────────────────────────────────────────
# Section 3 - Forward Pass
# ─────────────────────────────────────────────────────────────

def forward_pass(X, params):
    """Compute the forward pass through the two-layer network.

    Architecture:
        Z1 = X  @ W1.T + b1          # (N, n_hidden)
        A1 = tanh(Z1)                # (N, n_hidden)
        Z2 = A1 @ W2.T + b2          # (N, n_output)
        A2 = sigmoid(Z2)             # (N, n_output)

    Parameters
    ----------
    X      : np.ndarray shape (N, n_input)  - input batch
    params : dict with keys 'W1', 'b1', 'W2', 'b2'

    Returns
    -------
    A2    : np.ndarray shape (N, n_output) - output probabilities
    cache : dict with keys 'X', 'Z1', 'A1', 'Z2', 'A2'
            (store ALL intermediate values - needed for backprop)

    Hint: unpack params['W1'], params['b1'], etc. with
          W1, b1, W2, b2 = params['W1'], ...
    """
    raise NotImplementedError("TODO: implement forward_pass")


# ─────────────────────────────────────────────────────────────
# Section 4 - Loss
# ─────────────────────────────────────────────────────────────

def binary_cross_entropy(A2, y):
    """Binary cross-entropy loss averaged over the batch.

        L = -1/N * Σ [ y·log(ŷ) + (1-y)·log(1-ŷ) ]

    Parameters
    ----------
    A2 : np.ndarray shape (N, 1) - predicted probabilities
    y  : np.ndarray shape (N,) or (N, 1) - true binary labels

    Returns
    -------
    float - scalar loss value

    Hint: reshape y to (N, 1) with y.reshape(-1, 1) before computing.
    Clip A2 to [1e-12, 1-1e-12] to avoid log(0).
    Use np.mean over all samples.
    """
    raise NotImplementedError("TODO: implement binary_cross_entropy")


# ─────────────────────────────────────────────────────────────
# Section 5 - Backward Pass
# ─────────────────────────────────────────────────────────────

def backward_pass(params, cache, y):
    """Backpropagation: compute gradients of the loss w.r.t. all parameters.

    Step-by-step (N = batch size):

    1. Output layer error (combined sigmoid + BCE gradient - elegant result):
           dZ2 = A2 - y_col          shape (N, n_output)

    2. Output layer parameter gradients:
           dW2 = (1/N) * dZ2.T @ A1  shape (n_output, n_hidden)
           db2 = (1/N) * sum(dZ2, axis=0)  shape (n_output,)

    3. Propagate error back through hidden layer:
           dA1 = dZ2 @ W2            shape (N, n_hidden)
           dZ1 = dA1 * tanh_derivative(Z1)   (element-wise)  shape (N, n_hidden)

    4. Hidden layer parameter gradients:
           dW1 = (1/N) * dZ1.T @ X   shape (n_hidden, n_input)
           db1 = (1/N) * sum(dZ1, axis=0)  shape (n_hidden,)

    Parameters
    ----------
    params : dict - current parameters ('W1', 'b1', 'W2', 'b2')
    cache  : dict - saved from forward_pass ('X', 'Z1', 'A1', 'Z2', 'A2')
    y      : np.ndarray shape (N,) - true binary labels

    Returns
    -------
    grads : dict with keys 'dW1', 'db1', 'dW2', 'db2'

    Hint: unpack everything first:
        X, Z1, A1, Z2, A2 = cache['X'], cache['Z1'], ...
        W2 = params['W2']
        y_col = y.reshape(-1, 1)
    """
    raise NotImplementedError("TODO: implement backward_pass")


# ─────────────────────────────────────────────────────────────
# Section 6 - Parameter Update
# ─────────────────────────────────────────────────────────────

def update_params(params, grads, lr):
    """Gradient descent parameter update.

        W ← W - lr * dW
        b ← b - lr * db

    Parameters
    ----------
    params : dict - current parameters (modified IN PLACE)
    grads  : dict - gradients from backward_pass
    lr     : float - learning rate

    Returns
    -------
    params : dict - updated parameters (same dict, modified in place)

    Hint: iterate over ('W1', 'b1', 'W2', 'b2') and subtract lr * grads['d'+key].
    """
    raise NotImplementedError("TODO: implement update_params")


# ─────────────────────────────────────────────────────────────
# Section 7 - Training Loop (provided - do not modify)
# ─────────────────────────────────────────────────────────────

def train(X, y, n_hidden=8, lr=0.1, n_epochs=1000, seed=42, verbose=True):
    """Full training loop - uses the functions you implemented above.

    Parameters
    ----------
    X        : np.ndarray shape (N, n_input)
    y        : np.ndarray shape (N,)
    n_hidden : int   - number of hidden neurons
    lr       : float - learning rate
    n_epochs : int   - number of gradient descent steps
    seed     : int   - random seed
    verbose  : bool  - print loss every 200 epochs

    Returns
    -------
    params     : dict - trained parameters
    loss_curve : list of float - loss at each epoch
    """
    params = initialise_weights(X.shape[1], n_hidden, 1, seed=seed)
    loss_curve = []

    for epoch in range(n_epochs):
        A2, cache = forward_pass(X, params)
        loss = binary_cross_entropy(A2, y)
        loss_curve.append(loss)
        grads = backward_pass(params, cache, y)
        params = update_params(params, grads, lr)

        if verbose and (epoch % 200 == 0 or epoch == n_epochs - 1):
            print(f"  Epoch {epoch:>5d}  loss={loss:.4f}")

    return params, loss_curve


# ─────────────────────────────────────────────────────────────
# Section 8 - Prediction (provided - do not modify)
# ─────────────────────────────────────────────────────────────

def predict(X, params, threshold=0.5):
    """Predict binary class labels.

    Parameters
    ----------
    X         : np.ndarray shape (N, n_input)
    params    : dict - trained parameters
    threshold : float - decision threshold (default 0.5)

    Returns
    -------
    np.ndarray shape (N,) - predicted labels (0 or 1)
    """
    probs, _ = forward_pass(X, params)
    return (probs.ravel() >= threshold).astype(int)


def predict_proba(X, params):
    """Return output probabilities (before thresholding).

    Parameters
    ----------
    X      : np.ndarray shape (N, n_input)
    params : dict - trained parameters

    Returns
    -------
    np.ndarray shape (N,) - probabilities in (0, 1)
    """
    probs, _ = forward_pass(X, params)
    return probs.ravel()
