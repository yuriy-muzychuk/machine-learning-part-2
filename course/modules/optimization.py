"""
Optimization algorithms for neural network training.

This module provides optimization algorithms including SGD, SGD with momentum,
and Adam for training neural networks.

Dependencies:
    - numpy>=1.24.0
"""

import numpy as np
from typing import Dict, Tuple


def sgd(
    params: Dict[str, np.ndarray],
    grads: Dict[str, np.ndarray],
    learning_rate: float,
) -> Dict[str, np.ndarray]:
    """
    Stochastic Gradient Descent optimizer step.

    Parameters
    ----------
    params : dict
        Dictionary containing parameter arrays (weights, biases)
    grads : dict
        Dictionary containing gradient arrays (same keys as params)
    learning_rate : float
        Learning rate for parameter updates

    Returns
    -------
    updated_params : dict
        Dictionary with updated parameters (same structure as params)

    Note
    ----
    Updates parameters in-place: params[key] -= learning_rate * grads[key]
    """
    updated_params = {}
    for key in params:
        updated_params[key] = params[key] - learning_rate * grads[key]
    return updated_params


def sgd_momentum(
    params: Dict[str, np.ndarray],
    grads: Dict[str, np.ndarray],
    velocities: Dict[str, np.ndarray],
    learning_rate: float,
    momentum: float = 0.9,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    SGD with momentum optimizer step.

    Parameters
    ----------
    params : dict
        Dictionary containing parameter arrays
    grads : dict
        Dictionary containing gradient arrays
    velocities : dict
        Dictionary containing velocity arrays (initialized to zeros)
    learning_rate : float
        Learning rate for parameter updates
    momentum : float, optional
        Momentum coefficient (default: 0.9)

    Returns
    -------
    updated_params : dict
        Dictionary with updated parameters
    updated_velocities : dict
        Dictionary with updated velocities
    """
    updated_params = {}
    updated_velocities = {}

    for key in params:
        # Update velocity
        updated_velocities[key] = momentum * velocities[key] - learning_rate * grads[key]
        # Update parameters
        updated_params[key] = params[key] + updated_velocities[key]

    return updated_params, updated_velocities


def adam(
    params: Dict[str, np.ndarray],
    grads: Dict[str, np.ndarray],
    m: Dict[str, np.ndarray],
    v: Dict[str, np.ndarray],
    t: int,
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Adam optimizer step.

    Parameters
    ----------
    params : dict
        Dictionary containing parameter arrays
    grads : dict
        Dictionary containing gradient arrays
    m : dict
        Dictionary containing first moment estimates (initialized to zeros)
    v : dict
        Dictionary containing second moment estimates (initialized to zeros)
    t : int
        Time step (iteration number, starts at 1)
    learning_rate : float, optional
        Learning rate (default: 0.001)
    beta1 : float, optional
        Exponential decay rate for first moment (default: 0.9)
    beta2 : float, optional
        Exponential decay rate for second moment (default: 0.999)
    epsilon : float, optional
        Small constant for numerical stability (default: 1e-8)

    Returns
    -------
    updated_params : dict
        Dictionary with updated parameters
    updated_m : dict
        Dictionary with updated first moment estimates
    updated_v : dict
        Dictionary with updated second moment estimates
    """
    updated_params = {}
    updated_m = {}
    updated_v = {}

    for key in params:
        # Update biased first moment estimate
        updated_m[key] = beta1 * m[key] + (1 - beta1) * grads[key]
        # Update biased second raw moment estimate
        updated_v[key] = beta2 * v[key] + (1 - beta2) * (grads[key] ** 2)

        # Compute bias-corrected first moment estimate
        m_corrected = updated_m[key] / (1 - beta1**t)
        # Compute bias-corrected second raw moment estimate
        v_corrected = updated_v[key] / (1 - beta2**t)

        # Update parameters
        updated_params[key] = (
            params[key] - learning_rate * m_corrected / (np.sqrt(v_corrected) + epsilon)
        )

    return updated_params, updated_m, updated_v

