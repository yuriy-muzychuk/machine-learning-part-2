"""
Data preprocessing utilities for machine learning.

This module provides functions for data normalization and train-test splitting.

Dependencies:
    - numpy>=1.24.0
    - scikit-learn>=1.3.0
"""

import numpy as np
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split as sklearn_train_test_split


def normalize(
    X: np.ndarray, method: str = "standard"
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Normalizes input data.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    method : str, optional
        Normalization method. Options: 'standard' (zero mean, unit variance),
        'minmax' (scale to [0,1]) (default: 'standard')

    Returns
    -------
    X_normalized : np.ndarray
        Normalized data, same shape as X
    scaler_params : dict
        Parameters used for normalization (for inverse transform if needed)
        Contains 'mean' and 'std' for 'standard', or 'min' and 'max' for 'minmax'
    """
    if method == "standard":
        # Zero mean, unit variance
        mean = np.mean(X, axis=0, keepdims=True)
        std = np.std(X, axis=0, keepdims=True)
        # Avoid division by zero
        std = np.where(std == 0, 1.0, std)
        X_normalized = (X - mean) / std
        scaler_params = {"mean": mean, "std": std, "method": "standard"}

    elif method == "minmax":
        # Scale to [0, 1]
        min_val = np.min(X, axis=0, keepdims=True)
        max_val = np.max(X, axis=0, keepdims=True)
        # Avoid division by zero
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1.0, range_val)
        X_normalized = (X - min_val) / range_val
        scaler_params = {"min": min_val, "max": max_val, "method": "minmax"}

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return X_normalized, scaler_params


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits data into training and testing sets.

    Parameters
    ----------
    X : np.ndarray
        Input features, shape (n_samples, n_features)
    y : np.ndarray
        Target labels, shape (n_samples,) or (n_samples, n_classes)
    test_size : float, optional
        Proportion of dataset to include in test set (default: 0.2)
    random_state : int, optional
        Random seed for reproducibility (default: None)

    Returns
    -------
    X_train : np.ndarray
        Training features
    X_test : np.ndarray
        Testing features
    y_train : np.ndarray
        Training labels
    y_test : np.ndarray
        Testing labels

    Note
    ----
    Wrapper around sklearn.model_selection.train_test_split for consistency
    """
    return sklearn_train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

