"""
Reusable visualization functions for notebooks.

This module provides plotting functions for decision boundaries, training curves,
and latent space visualizations.

Dependencies:
    - numpy>=1.24.0
    - matplotlib>=3.7.0
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple, Union

from plot_style import COLORBLIND_PALETTE


def plot_decision_boundary(
    X: np.ndarray,
    y: np.ndarray,
    predict_fn: Callable,
    resolution: int = 100,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots decision boundary for binary classification.

    Parameters
    ----------
    X : np.ndarray
        Input features, shape (n_samples, 2) (2D data only)
    y : np.ndarray
        True labels, shape (n_samples,)
    predict_fn : callable
        Prediction function that takes X and returns predictions
    resolution : int, optional
        Grid resolution for decision boundary (default: 100)

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object for further customization
    ax : matplotlib.axes.Axes
        Axes object for further customization
    """
    if X.shape[1] != 2:
        raise ValueError("plot_decision_boundary only supports 2D data")

    # Create a mesh
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )

    # Predict on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_fn(mesh_points)
    Z = Z.reshape(xx.shape)

    # Plot
    from matplotlib.colors import ListedColormap
    cmap_boundary = ListedColormap([COLORBLIND_PALETTE[0], COLORBLIND_PALETTE[3]])
    cmap_points   = ListedColormap([COLORBLIND_PALETTE[0], COLORBLIND_PALETTE[3]])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_boundary)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_points, edgecolors="k")
    ax.set_xlabel("Ознака 1")
    ax.set_ylabel("Ознака 2")
    ax.set_title("Межа рішення")
    plt.colorbar(scatter, ax=ax)

    return fig, ax


def plot_training_curves(
    train_losses: Union[list, np.ndarray],
    val_losses: Optional[Union[list, np.ndarray]] = None,
    train_accuracies: Optional[Union[list, np.ndarray]] = None,
    val_accuracies: Optional[Union[list, np.ndarray]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plots training curves (loss and/or accuracy over epochs).

    Parameters
    ----------
    train_losses : list or np.ndarray
        Training loss values per epoch
    val_losses : list or np.ndarray, optional
        Validation loss values per epoch
    train_accuracies : list or np.ndarray, optional
        Training accuracy values per epoch
    val_accuracies : list or np.ndarray, optional
        Validation accuracy values per epoch

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object for further customization
    axes : np.ndarray
        Array of axes objects
    """
    train_losses = np.array(train_losses)
    n_plots = 1
    if train_accuracies is not None:
        n_plots = 2

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    if n_plots == 1:
        axes = [axes]

    # Plot losses
    ax = axes[0]
    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Навчання", marker="o", color=COLORBLIND_PALETTE[0])
    if val_losses is not None:
        val_losses = np.array(val_losses)
        ax.plot(epochs, val_losses, label="Валідація", marker="s", color=COLORBLIND_PALETTE[1])
    ax.set_xlabel("Епоха")
    ax.set_ylabel("Функція втрат")
    ax.set_title("Крива навчання")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot accuracies if provided
    if train_accuracies is not None:
        ax = axes[1]
        train_accuracies = np.array(train_accuracies)
        ax.plot(epochs, train_accuracies, label="Навчання", marker="o", color=COLORBLIND_PALETTE[0])
        if val_accuracies is not None:
            val_accuracies = np.array(val_accuracies)
            ax.plot(epochs, val_accuracies, label="Валідація", marker="s", color=COLORBLIND_PALETTE[1])
        ax.set_xlabel("Епоха")
        ax.set_ylabel("Точність")
        ax.set_title("Точність на навчальній та валідаційній вибірках")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


def plot_latent_space(
    Z: np.ndarray,
    y: Optional[np.ndarray] = None,
    title: str = "Latent Space Visualization",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots 2D latent space visualization.

    Parameters
    ----------
    Z : np.ndarray
        Latent representations, shape (n_samples, 2) (2D only) or
        (n_samples, n_dims) (will use PCA to reduce to 2D)
    y : np.ndarray, optional
        Labels for coloring points, shape (n_samples,)
    title : str, optional
        Plot title (default: 'Latent Space Visualization')

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object for further customization
    ax : matplotlib.axes.Axes
        Axes object for further customization
    """
    # If Z is not 2D, reduce using PCA
    if Z.shape[1] != 2:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        Z = pca.fit_transform(Z)

    fig, ax = plt.subplots(figsize=(10, 8))

    if y is not None:
        scatter = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap=plt.cm.viridis, alpha=0.6)
        plt.colorbar(scatter, ax=ax, label="Клас")
    else:
        ax.scatter(Z[:, 0], Z[:, 1], alpha=0.6, color=COLORBLIND_PALETTE[0])

    ax.set_xlabel("Латентний вимір 1")
    ax.set_ylabel("Латентний вимір 2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return fig, ax

