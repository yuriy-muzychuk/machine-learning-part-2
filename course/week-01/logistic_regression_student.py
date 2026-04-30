"""
Logistic Regression implementation from scratch using NumPy.

This is the STUDENT version - complete the TODO sections to make it work.
Functions are imported and used in the lab notebook.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def sigmoid(z):
    """Compute numerically stable sigmoid function.

    Parameters
    ----------
    z : np.ndarray or float
        Linear combination of inputs.

    Returns
    -------
    np.ndarray
        Values in range (0, 1).
    """
    # TODO: Implement sigmoid
    # Hint: Use np.clip(z, -500, 500) to prevent overflow, then apply:
    #   sigma(z) = 1 / (1 + exp(-z))
    raise NotImplementedError("sigmoid is not implemented yet")


def logistic_loss(y_true, y_pred):
    """Compute binary cross-entropy loss.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Ground truth binary labels {0, 1}.
    y_pred : np.ndarray, shape (n_samples,)
        Predicted probabilities in (0, 1).

    Returns
    -------
    float
        Mean binary cross-entropy loss.
    """
    # TODO: Implement binary cross-entropy loss
    # Hint: Clip y_pred to [1e-15, 1 - 1e-15] to avoid log(0)
    # Formula: L = -mean( y * log(y_pred) + (1-y) * log(1-y_pred) )
    raise NotImplementedError("logistic_loss is not implemented yet")


def predict(X, w, b):
    """Predict binary class labels.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    w : np.ndarray, shape (n_features,)
    b : float

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Predicted labels {0, 1}.
    """
    # TODO: Implement predict
    # Hint: Compute z = X @ w + b, apply sigmoid, then threshold at 0.5
    raise NotImplementedError("predict is not implemented yet")


def compute_accuracy(y_true, y_pred):
    """Compute classification accuracy.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
    y_pred : np.ndarray, shape (n_samples,)

    Returns
    -------
    float
        Fraction of correctly classified samples.
    """
    # TODO: Implement accuracy
    # Hint: np.mean(y_true == y_pred)
    raise NotImplementedError("compute_accuracy is not implemented yet")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_logistic_regression(X, y, learning_rate=0.01, epochs=1000):
    """Train logistic regression using batch gradient descent.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    learning_rate : float
    epochs : int

    Returns
    -------
    w : np.ndarray, shape (n_features,)
    b : float
    losses : list of float
    """
    n_samples, n_features = X.shape

    # TODO: Initialize weights w to zeros and bias b to 0.0
    w = None
    b = None

    losses = []

    for epoch in range(epochs):
        # TODO: Forward pass - compute z = X @ w + b, then y_pred = sigmoid(z)
        y_pred = None

        # TODO: Compute loss using logistic_loss and append to losses
        loss = None
        losses.append(loss)

        # Backward pass - gradients are provided
        dw = np.dot(X.T, (y_pred - y)) / n_samples
        db = np.mean(y_pred - y)

        # TODO: Update parameters
        # w = w - learning_rate * dw
        # b = b - learning_rate * db

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return w, b, losses


# Modified training function that tracks weight history
def train_logistic_regression_with_weights(X, y, learning_rate=0.01, epochs=1000, track_weights=False):
    """Train logistic regression model with optional weight tracking."""
    n_samples, n_features = X.shape
    
    # Initialize weights and bias
    w = np.zeros(n_features)
    b = 0.0
    
    losses = []
    weight_history = [] if track_weights else None
    
    for epoch in range(epochs):
        # Forward pass
        
        y_pred = None
        
        # Compute loss
        loss = None
        losses.append(loss)
        
        if track_weights:
            weight_history.append({
                'w': w.copy(),
                'b': b,
                'weight_mag': np.linalg.norm(w)
            })
        
        # Backward pass (gradients)
        dw = np.dot(X.T, (y_pred - y)) / n_samples
        db = np.mean(y_pred - y)
        
        # Update parameters
        w -= None
        b -= None
        
        # Check for NaN/Inf (instability)
        if np.isnan(w).any() or np.isinf(w).any() or np.isnan(b) or np.isinf(b):
            print(f"Warning: NaN/Inf detected at epoch {epoch} for LR={learning_rate}")
            break
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}, ||w||: {np.linalg.norm(w):.4f}")
    
    return w, b, losses, weight_history