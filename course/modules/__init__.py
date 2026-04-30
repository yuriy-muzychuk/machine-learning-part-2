"""
Course modules package.

This package contains reusable Python modules for the machine learning course.
"""

from .neural_networks import forward_pass, backward_pass, compute_loss
from .optimization import sgd, sgd_momentum, adam
from .preprocessing import normalize, train_test_split
from .visualization import plot_decision_boundary, plot_training_curves, plot_latent_space
from .evaluation import compute_accuracy, compute_confusion_matrix, plot_confusion_matrix
from .plot_style import setup_plot_style, COLORBLIND_PALETTE
from .pca_svd_solution import PCA, TruncatedSVD
from .density_estimation_solution import KDE, GMM

__all__ = [
    # Neural networks
    "forward_pass",
    "backward_pass",
    "compute_loss",
    # Optimization
    "sgd",
    "sgd_momentum",
    "adam",
    # Preprocessing
    "normalize",
    "train_test_split",
    # Visualization
    "plot_decision_boundary",
    "plot_training_curves",
    "plot_latent_space",
    # Evaluation
    "compute_accuracy",
    "compute_confusion_matrix",
    "plot_confusion_matrix",
    # Plot style
    "setup_plot_style",
    "COLORBLIND_PALETTE",
    # PCA / SVD
    "PCA",
    "TruncatedSVD",
    # Density estimation
    "KDE",
    "GMM",
]

