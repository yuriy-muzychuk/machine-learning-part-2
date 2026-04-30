"""
Neural network building blocks for forward/backward passes and activation functions.

This module provides core functions for implementing neural networks from scratch,
including forward propagation, backward propagation, and loss computation.

Dependencies:
    - numpy>=1.24.0
"""

import numpy as np
from typing import Tuple


def forward_pass(
    X: np.ndarray, W: np.ndarray, b: np.ndarray, activation: str = "relu"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs forward pass through a single layer.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    W : np.ndarray
        Weight matrix of shape (n_features, n_hidden)
    b : np.ndarray
        Bias vector of shape (n_hidden,)
    activation : str, optional
        Activation function name. Options: 'relu', 'sigmoid', 'tanh', 'linear'
        (default: 'relu')

    Returns
    -------
    Z : np.ndarray
        Linear output before activation, shape (n_samples, n_hidden)
    A : np.ndarray
        Activated output, shape (n_samples, n_hidden)

    Raises
    ------
    ValueError
        If activation function name is not recognized
    """
    # Linear transformation
    Z = np.dot(X, W) + b

    # Apply activation function
    if activation == "relu":
        A = np.maximum(0, Z)
    elif activation == "sigmoid":
        A = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))  # Clip to avoid overflow
    elif activation == "tanh":
        A = np.tanh(Z)
    elif activation == "linear":
        A = Z
    else:
        raise ValueError(f"Unknown activation function: {activation}")

    return Z, A


def backward_pass(
    dA: np.ndarray, Z: np.ndarray, activation: str = "relu"
) -> np.ndarray:
    """
    Performs backward pass to compute gradients for a single layer.

    Parameters
    ----------
    dA : np.ndarray
        Gradient of loss with respect to activated output, shape (n_samples, n_hidden)
    Z : np.ndarray
        Linear output from forward pass (before activation), shape (n_samples, n_hidden)
    activation : str, optional
        Activation function name. Options: 'relu', 'sigmoid', 'tanh', 'linear'
        (default: 'relu')

    Returns
    -------
    dZ : np.ndarray
        Gradient of loss with respect to linear output, shape (n_samples, n_hidden)

    Raises
    ------
    ValueError
        If activation function name is not recognized
    """
    if activation == "relu":
        dZ = dA * (Z > 0).astype(float)
    elif activation == "sigmoid":
        A = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
        dZ = dA * A * (1 - A)
    elif activation == "tanh":
        A = np.tanh(Z)
        dZ = dA * (1 - A**2)
    elif activation == "linear":
        dZ = dA
    else:
        raise ValueError(f"Unknown activation function: {activation}")

    return dZ


def compute_loss(
    y_true: np.ndarray, y_pred: np.ndarray, loss_type: str = "cross_entropy"
) -> float:
    """
    Computes loss between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        True labels, shape (n_samples,) or (n_samples, n_classes)
    y_pred : np.ndarray
        Predicted probabilities/logits, shape (n_samples, n_classes)
    loss_type : str, optional
        Loss function name. Options: 'cross_entropy', 'mse', 'binary_cross_entropy'
        (default: 'cross_entropy')

    Returns
    -------
    loss : float
        Computed loss value

    Raises
    ------
    ValueError
        If loss function name is not recognized or shapes are incompatible
    """
    n_samples = y_true.shape[0]

    if loss_type == "cross_entropy":
        # Softmax and cross-entropy for multi-class classification
        # Clip predictions to avoid numerical issues
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        # Apply softmax
        exp_pred = np.exp(y_pred_clipped - np.max(y_pred_clipped, axis=1, keepdims=True))
        softmax_pred = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)

        # Convert y_true to one-hot if needed
        if y_true.ndim == 1:
            n_classes = y_pred.shape[1]
            y_true_onehot = np.eye(n_classes)[y_true]
        else:
            y_true_onehot = y_true

        # Cross-entropy loss
        loss = -np.sum(y_true_onehot * np.log(softmax_pred)) / n_samples

    elif loss_type == "binary_cross_entropy":
        # Binary cross-entropy
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        loss = -np.mean(
            y_true * np.log(y_pred_clipped)
            + (1 - y_true) * np.log(1 - y_pred_clipped)
        )

    elif loss_type == "mse":
        # Mean squared error
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        loss = np.mean((y_true - y_pred) ** 2)

    else:
        raise ValueError(f"Unknown loss function: {loss_type}")

    return float(loss)

