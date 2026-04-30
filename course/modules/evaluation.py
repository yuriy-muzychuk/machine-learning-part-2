"""
Model evaluation metrics and utilities.

This module provides functions for computing accuracy, confusion matrices,
and plotting evaluation results.

Dependencies:
    - numpy>=1.24.0
    - matplotlib>=3.7.0
    - scikit-learn>=1.3.0
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes classification accuracy.

    Parameters
    ----------
    y_true : np.ndarray
        True labels, shape (n_samples,)
    y_pred : np.ndarray
        Predicted labels, shape (n_samples,)

    Returns
    -------
    accuracy : float
        Accuracy score between 0 and 1
    """
    return float(np.mean(y_true == y_pred))


def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: Optional[int] = None
) -> np.ndarray:
    """
    Computes confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        True labels, shape (n_samples,)
    y_pred : np.ndarray
        Predicted labels, shape (n_samples,)
    n_classes : int, optional
        Number of classes (inferred from data if None)

    Returns
    -------
    confusion_matrix : np.ndarray
        Confusion matrix of shape (n_classes, n_classes)
    """
    return sklearn_confusion_matrix(y_true, y_pred, labels=n_classes)


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[list] = None,
    normalize: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots confusion matrix as heatmap.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        Confusion matrix of shape (n_classes, n_classes)
    class_names : list, optional
        List of class names for labeling
    normalize : bool, optional
        Whether to normalize confusion matrix (default: False)

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object for further customization
    ax : matplotlib.axes.Axes
        Axes object for further customization
    """
    if normalize:
        cm = confusion_matrix.astype("float") / confusion_matrix.sum(axis=1)[:, np.newaxis]
        title = "Normalized Confusion Matrix"
    else:
        cm = confusion_matrix
        title = "Confusion Matrix"

    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Set ticks and labels
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j,
                i,
                format(cm[i, j], ".2f" if normalize else "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    return fig, ax

