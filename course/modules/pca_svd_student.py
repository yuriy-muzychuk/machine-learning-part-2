"""
pca_svd_student.py - PCA and Truncated SVD from Scratch (Student Version)
=========================================================================
Week 7 Lab: Implement Principal Component Analysis and Truncated SVD
using only NumPy. Apply both methods to synthetic data, a face image
dataset, and signal denoising.

Instructions
------------
Implement every method marked with
    raise NotImplementedError("TODO: ...")

Do NOT import anything beyond what is already imported.
Do NOT change method signatures or the __init__ constructors.
The notebook tests each method in order - implement them top to bottom.

Testing
-------
Run the notebook cells in order. Each Part tests one group of methods.
The equivalence check in Part 3 will automatically verify that your
PCA and TruncatedSVD produce consistent results on centred data.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _center(X):
    """Subtract column means from X. Returns (X_centered, mean)."""
    mean = X.mean(axis=0)
    return X - mean, mean


# ---------------------------------------------------------------------------
# PCA  (via covariance matrix eigen-decomposition)
# ---------------------------------------------------------------------------

class PCA:
    """Principal Component Analysis via covariance matrix eigen-decomposition.

    Standardises input to zero mean (but NOT unit variance) before computing
    the covariance matrix. Scaling to unit variance is the caller's
    responsibility (use StandardScaler when needed).

    Parameters
    ----------
    n_components : int or None
        Number of principal components to keep.
        If None, all components are kept.

    Attributes (set after fit)
    --------------------------
    components_          : ndarray (n_components, n_features)
        Principal directions (eigenvectors), rows sorted by descending eigenvalue.
    explained_variance_  : ndarray (n_components,)
        Eigenvalues of the covariance matrix (variance along each PC).
    explained_variance_ratio_ : ndarray (n_components,)
        Fraction of total variance explained by each PC.
    mean_                : ndarray (n_features,)
        Per-feature mean of the training data.
    n_components_        : int
        Actual number of components stored.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

        self.components_               = None
        self.explained_variance_       = None
        self.explained_variance_ratio_ = None
        self.mean_                     = None
        self.n_components_             = None

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def fit(self, X):
        """Compute principal components from training data X.

        Algorithm
        ---------
        1. Center X:             X_c = X - mean(X, axis=0)
        2. Covariance matrix:    C   = X_c.T @ X_c / (n - 1)
        3. Eigen-decomposition:  C v_i = lambda_i v_i
                                 use np.linalg.eigh (stable for symmetric C)
        4. Sort eigenvalues and eigenvectors in DESCENDING order
        5. Keep the top n_components eigenvectors
           - If self.n_components is None, keep all (min(n_samples, n_features))
        6. Store:
           - self.mean_                     = column means of X
           - self.components_               = top-k eigenvectors as ROWS, shape (k, n_features)
           - self.explained_variance_       = top-k eigenvalues,    shape (k,)
           - self.explained_variance_ratio_ = eigenvalues / sum(all eigenvalues), shape (k,)
           - self.n_components_             = k

        Parameters
        ----------
        X : ndarray shape (n_samples, n_features)

        Returns
        -------
        self

        Hint
        ----
        np.linalg.eigh returns eigenvalues in ASCENDING order - reverse them.
        Eigenvectors are columns of the returned matrix, not rows.
        """
        raise NotImplementedError("TODO: implement fit")

    def transform(self, X):
        """Project X onto the principal subspace.

        Z = (X - mean_) @ components_.T

        Parameters
        ----------
        X : ndarray shape (n_samples, n_features)

        Returns
        -------
        Z : ndarray shape (n_samples, n_components_)

        Hint
        ----
        Subtract self.mean_ before multiplying by self.components_.T
        """
        raise NotImplementedError("TODO: implement transform")

    def fit_transform(self, X):
        """Fit and return the projection in one call.

        Equivalent to self.fit(X).transform(X).
        """
        raise NotImplementedError("TODO: implement fit_transform")

    def inverse_transform(self, Z):
        """Reconstruct approximate X from low-dimensional representation Z.

        X_hat = Z @ components_ + mean_

        Parameters
        ----------
        Z : ndarray shape (n_samples, n_components_)

        Returns
        -------
        X_hat : ndarray shape (n_samples, n_features)

        Hint
        ----
        This is the reverse of transform: rotate back with components_ (not .T)
        and add the mean back.
        """
        raise NotImplementedError("TODO: implement inverse_transform")

    def reconstruction_error(self, X):
        """Mean squared reconstruction error ||X - X_hat||^2 / (n * d).

        Parameters
        ----------
        X : ndarray shape (n_samples, n_features)

        Returns
        -------
        float

        Hint
        ----
        Use transform then inverse_transform, then compute np.mean of squared
        element-wise differences.
        """
        raise NotImplementedError("TODO: implement reconstruction_error")


# ---------------------------------------------------------------------------
# TruncatedSVD  (directly via np.linalg.svd - no centering)
# ---------------------------------------------------------------------------

class TruncatedSVD:
    """Dimensionality reduction via Truncated Singular Value Decomposition.

    Unlike PCA, TruncatedSVD does NOT center the data before factorisation.
    This makes it suitable for sparse matrices (e.g. TF-IDF) where centering
    would destroy sparsity.  For dense centered data it is equivalent to PCA.

    The approximation uses the top-k singular triplets:
        X ≈ U_k  Σ_k  V_k^T

    Parameters
    ----------
    n_components : int
        Number of singular value / vector triplets to keep.

    Attributes (set after fit)
    --------------------------
    components_       : ndarray (n_components, n_features)  - V_k^T rows
    singular_values_  : ndarray (n_components,)             - σ_1 ≥ … ≥ σ_k
    explained_variance_ratio_ : ndarray (n_components,)
    """

    def __init__(self, n_components=2):
        self.n_components = n_components

        self.components_               = None
        self.singular_values_          = None
        self.explained_variance_ratio_ = None

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def fit(self, X):
        """Compute the truncated SVD of X.

        Algorithm
        ---------
        1. Economy SVD: U, s, Vt = np.linalg.svd(X, full_matrices=False)
           - U  shape (n_samples, r)
           - s  shape (r,)            singular values, already descending
           - Vt shape (r, n_features) right singular vectors as rows
        2. Keep only the top n_components singular triplets
        3. Store:
           - self.singular_values_          = s[:k]
           - self.components_               = Vt[:k]   (k, n_features)
           - self.explained_variance_ratio_ = s[:k]^2 / sum(s^2)

        Parameters
        ----------
        X : ndarray shape (n_samples, n_features)

        Returns
        -------
        self

        Hint
        ----
        np.linalg.svd already returns singular values in descending order,
        so no sorting is needed.
        """
        raise NotImplementedError("TODO: implement fit")

    def transform(self, X):
        """Project X onto the top-k right singular vectors.

        Z = X @ V_k  =  X @ components_.T

        Parameters
        ----------
        X : ndarray shape (n_samples, n_features)

        Returns
        -------
        Z : ndarray shape (n_samples, n_components)

        Hint
        ----
        Multiply X by self.components_.T  (note: no mean subtraction here)
        """
        raise NotImplementedError("TODO: implement transform")

    def fit_transform(self, X):
        """Fit and return the projection in one call."""
        raise NotImplementedError("TODO: implement fit_transform")

    def inverse_transform(self, Z):
        """Reconstruct approximate X from the low-dimensional representation.

        X_hat = Z @ components_

        Parameters
        ----------
        Z : ndarray shape (n_samples, n_components)

        Returns
        -------
        X_hat : ndarray shape (n_samples, n_features)
        """
        raise NotImplementedError("TODO: implement inverse_transform")

    def reconstruction_error(self, X):
        """Mean squared reconstruction error.

        By the Eckart-Young theorem this equals sum(s[k:]^2) / (n * d),
        but we compute it directly for verification.

        Parameters
        ----------
        X : ndarray shape (n_samples, n_features)

        Returns
        -------
        float
        """
        raise NotImplementedError("TODO: implement reconstruction_error")
