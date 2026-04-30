"""
Logistic Regression module with BGD, SGD, MBGD, L1/L2 regularization,
early stopping, and optimization diagnostics.

This is the STUDENT version — complete the TODO sections to make it work.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def sigmoid(z):
    """Numerically stable sigmoid function."""
    # TODO: implement numerically stable sigmoid
    # Hint: use np.where to handle positive and negative z separately
    # For z >= 0: 1 / (1 + exp(-z))
    # For z <  0: exp(z) / (1 + exp(z))
    raise NotImplementedError("sigmoid is not implemented yet")


def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """Binary cross-entropy loss (without regularization)."""
    # TODO: clip y_pred to [eps, 1-eps] to avoid log(0), then compute:
    # L = -mean( y * log(y_pred) + (1-y) * log(1-y_pred) )
    raise NotImplementedError("binary_cross_entropy is not implemented yet")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LogisticRegression:
    """
    Logistic Regression classifier supporting multiple optimization strategies
    and regularization schemes.

    Parameters
    ----------
    learning_rate : float
        Step size for gradient updates.
    n_iterations : int
        Maximum number of passes over the training data (epochs).
    optimizer : str
        One of 'bgd' (Batch GD), 'sgd' (Stochastic GD), 'mbgd' (Mini-Batch GD).
    batch_size : int
        Number of samples per mini-batch (used only when optimizer='mbgd').
    penalty : str or None
        Regularization type: 'l2', 'l1', or None.
    lambda_ : float
        Regularization strength (λ).
    early_stopping : bool
        Whether to use early stopping based on validation loss.
    validation_fraction : float
        Fraction of training data to reserve for validation (early stopping).
    n_iter_no_change : int
        Number of iterations with no improvement before stopping.
    tol : float
        Minimum improvement in validation loss to count as progress.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        learning_rate=0.1,
        n_iterations=1000,
        optimizer="bgd",
        batch_size=32,
        penalty="l2",
        lambda_=0.01,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-4,
        random_state=None,
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.optimizer = optimizer.lower()
        self.batch_size = batch_size
        self.penalty = penalty
        self.lambda_ = lambda_
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.random_state = random_state

        # Attributes set during fit
        self.weights_ = None
        self.bias_ = None
        self.train_losses_ = []
        self.val_losses_ = []
        self.grad_norms_ = []
        self.n_iter_ = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_params(self, n_features):
        rng = np.random.default_rng(self.random_state)
        self.weights_ = rng.normal(0, 0.01, size=n_features)
        self.bias_ = 0.0

    def _forward(self, X):
        """Compute predicted probabilities."""
        # TODO: compute linear combination z = X @ w + b, then apply sigmoid
        raise NotImplementedError("_forward is not implemented yet")

    def _compute_loss(self, X, y):
        """Cross-entropy loss + regularization penalty."""
        y_pred = self._forward(X)
        loss = binary_cross_entropy(y, y_pred)

        # TODO: add regularization term to loss
        # If self.penalty == 'l2': add (lambda_ / 2) * sum(w^2)
        # If self.penalty == 'l1': add lambda_ * sum(|w|)
        # Note: bias is NOT regularized

        return loss

    def _compute_gradients(self, X, y):
        """
        Compute gradients of the loss w.r.t. weights and bias.

        Formulas (m = batch size):
            error  = y_hat - y
            dL/dw  = (1/m) X^T error  +  regularization gradient
            dL/db  = (1/m) sum(error)
        """
        m = X.shape[0]
        y_pred = self._forward(X)
        error = y_pred - y

        # TODO: compute dw and db from error
        dw = None  # replace with correct expression
        db = None  # replace with correct expression

        # TODO: add regularization gradient to dw
        # If self.penalty == 'l2': dw += lambda_ * w
        # If self.penalty == 'l1': dw += lambda_ * sign(w)

        return dw, db

    def _split_validation(self, X, y):
        """Split off a validation set for early stopping."""
        rng = np.random.default_rng(self.random_state)
        n_val = max(1, int(len(y) * self.validation_fraction))
        idx = rng.permutation(len(y))
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        return X[train_idx], X[val_idx], y[train_idx], y[val_idx]

    # ------------------------------------------------------------------
    # Optimization loops
    # ------------------------------------------------------------------

    def _bgd_step(self, X, y):
        """
        Single Batch Gradient Descent update.
        Uses the ENTIRE dataset to compute one gradient step.
        """
        # TODO: compute gradients on the full dataset and update weights and bias
        # self.weights_ -= learning_rate * dw
        # self.bias_    -= learning_rate * db
        raise NotImplementedError("_bgd_step is not implemented yet")

    def _sgd_step(self, X, y, rng):
        """
        Single Stochastic Gradient Descent update.
        Uses ONE randomly selected sample per update.
        """
        # TODO: sample a random index, compute gradients on that single sample,
        # and update weights and bias
        raise NotImplementedError("_sgd_step is not implemented yet")

    def _mbgd_epoch(self, X, y, rng):
        """
        One epoch of Mini-Batch Gradient Descent.
        Shuffles data, then iterates over mini-batches of size self.batch_size.
        """
        # TODO: shuffle X and y using rng.permutation, then loop over mini-batches
        # For each mini-batch [start : start + batch_size]:
        #   - compute gradients
        #   - update weights and bias
        # Return the last dw (used for gradient norm diagnostics)
        raise NotImplementedError("_mbgd_epoch is not implemented yet")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Train the logistic regression model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)  — binary labels {0, 1}
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        self._init_params(X.shape[1])
        self.train_losses_ = []
        self.val_losses_ = []
        self.grad_norms_ = []

        rng = np.random.default_rng(self.random_state)

        # Split validation set if early stopping is requested
        if self.early_stopping:
            X_train, X_val, y_train, y_val = self._split_validation(X, y)
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        best_val_loss = np.inf
        no_improve_count = 0

        for iteration in range(self.n_iterations):
            # --- gradient step ---
            if self.optimizer == "bgd":
                dw = self._bgd_step(X_train, y_train)
            elif self.optimizer == "sgd":
                dw = self._sgd_step(X_train, y_train, rng)
            elif self.optimizer == "mbgd":
                dw = self._mbgd_epoch(X_train, y_train, rng)
            else:
                raise ValueError(f"Unknown optimizer: {self.optimizer!r}. "
                                 "Choose from 'bgd', 'sgd', 'mbgd'.")

            # --- record diagnostics ---
            train_loss = self._compute_loss(X_train, y_train)
            self.train_losses_.append(train_loss)
            self.grad_norms_.append(np.linalg.norm(dw))

            # TODO: implement early stopping below
            # Steps:
            # 1. If self.early_stopping is True, compute val_loss on X_val / y_val
            # 2. Append val_loss to self.val_losses_
            # 3. If val_loss improved by more than self.tol compared to best_val_loss:
            #       - update best_val_loss
            #       - reset no_improve_count to 0
            #       - save a copy of current weights and bias (best_weights, best_bias)
            #    Else:
            #       - increment no_improve_count
            # 4. If no_improve_count >= self.n_iter_no_change:
            #       - restore weights and bias from best_weights / best_bias
            #       - set self.n_iter_ = iteration + 1
            #       - break the loop

            self.n_iter_ = iteration + 1

        return self

    def predict_proba(self, X):
        """Return predicted probabilities for class 1."""
        X = np.array(X, dtype=float)
        # TODO: call _forward and return result
        raise NotImplementedError("predict_proba is not implemented yet")

    def predict(self, X, threshold=0.5):
        """Return binary predictions (0 or 1)."""
        # TODO: call predict_proba, apply threshold, return int array
        raise NotImplementedError("predict is not implemented yet")

    def get_diagnostics(self):
        """
        Return a dict with recorded training history for plotting.

        Keys: 'train_losses', 'val_losses', 'grad_norms', 'n_iter'
        """
        return {
            "train_losses": self.train_losses_,
            "val_losses": self.val_losses_,
            "grad_norms": self.grad_norms_,
            "n_iter": self.n_iter_,
        }
