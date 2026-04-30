"""
nn_multiclass_student.py - Multi-Layer Neural Network (Student Version)
========================================================================
Week 4 Lab: Extend the two-layer network from Week 3 into a general
multi-layer neural network that supports:
  • Arbitrary depth (variable number of hidden layers)
  • Multiple activation functions: ReLU, sigmoid, tanh, softmax
  • Multiclass classification via softmax + categorical cross-entropy
  • Binary classification (single output with sigmoid)

Instructions
------------
Implement every method marked with
    raise NotImplementedError("TODO: ...")

Do NOT import anything beyond what is already imported.
Do NOT change method signatures or the __init__ constructor.
The provided methods (fit, predict, predict_proba) depend on your
implementations - do not modify them.

Architecture example:
    NeuralNetwork(layer_sizes=[2, 16, 8, 3],
                  activations=['relu', 'relu', 'softmax'])

    Input(2) → Dense(16, ReLU) → Dense(8, ReLU) → Dense(3, Softmax)

Testing
-------
Run the notebook cells in order. Each Part tests one group of methods.
The gradient check in Part 5 will catch most backprop bugs automatically.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 - Standalone Activation Functions (used outside the class too)
# ─────────────────────────────────────────────────────────────────────────────

def relu(z):
    """ReLU activation: max(0, z), element-wise.

    Parameters
    ----------
    z : np.ndarray, any shape

    Returns
    -------
    np.ndarray, same shape - negative values clamped to 0

    Hint: np.maximum(0, z)
    """
    raise NotImplementedError("TODO: implement relu")


def relu_derivative(z):
    """Derivative of ReLU: 1 where z > 0, else 0.

    Parameters
    ----------
    z : np.ndarray, any shape - PRE-ACTIVATION values

    Returns
    -------
    np.ndarray, same shape, dtype float

    Hint: (z > 0).astype(float)
    """
    raise NotImplementedError("TODO: implement relu_derivative")


def softmax(z):
    """Softmax activation (numerically stable), applied row-wise.

        softmax(z_i) = exp(z_i - max(z)) / sum(exp(z - max(z)))

    The max subtraction (log-sum-exp trick) prevents overflow.

    Parameters
    ----------
    z : np.ndarray shape (N, K) - K is the number of classes

    Returns
    -------
    np.ndarray shape (N, K) - each row sums to 1

    Hint:
        shifted = z - z.max(axis=1, keepdims=True)
        exps    = np.exp(shifted)
        return  exps / exps.sum(axis=1, keepdims=True)
    """
    raise NotImplementedError("TODO: implement softmax")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 - NeuralNetwork Class
# ─────────────────────────────────────────────────────────────────────────────

class NeuralNetwork:
    """General multi-layer neural network for classification.

    Supports binary and multiclass targets.  The output activation
    determines the loss:
        • 'softmax'  → categorical cross-entropy (multiclass)
        • 'sigmoid'  → binary cross-entropy      (binary)

    Parameters
    ----------
    layer_sizes : list of int
        Number of units in each layer including input and output.
        Example: [n_features, 64, 32, n_classes]
    activations : list of str
        Activation for each layer except the input. Length must be
        len(layer_sizes) - 1. Supported values: 'relu', 'tanh',
        'sigmoid', 'softmax'.
    random_state : int
        Seed for reproducible weight initialisation.

    Attributes (set after fit)
    --------------------------
    params_     : dict  - weights W1,b1, W2,b2, … indexed by layer
    loss_curve_ : list  - training loss recorded each epoch
    """

    def __init__(self, layer_sizes, activations, random_state=42):
        assert len(activations) == len(layer_sizes) - 1, (
            "len(activations) must equal len(layer_sizes) - 1"
        )
        self.layer_sizes   = layer_sizes
        self.activations   = activations   # e.g. ['relu', 'relu', 'softmax']
        self.random_state  = random_state
        self.params_       = {}
        self.loss_curve_   = []

    # ── Private helpers ───────────────────────────────────────────────────────

    def _apply_activation(self, z, name):
        """Apply activation function by name to pre-activation z.

        Parameters
        ----------
        z    : np.ndarray
        name : str - one of 'relu', 'tanh', 'sigmoid', 'softmax'

        Returns
        -------
        np.ndarray, same shape as z (except softmax preserves (N, K))
        """
        if name == 'relu':
            return relu(z)
        elif name == 'tanh':
            return np.tanh(z)
        elif name == 'sigmoid':
            z_clipped = np.clip(z, -500, 500)
            return 1.0 / (1.0 + np.exp(-z_clipped))
        elif name == 'softmax':
            return softmax(z)
        else:
            raise ValueError(f"Unknown activation: '{name}'")

    def _activation_derivative(self, z, name):
        """Derivative of activation by name w.r.t. PRE-ACTIVATION z.

        Note: for 'softmax' this method is NOT called during backprop -
        the softmax + cross-entropy gradient simplifies to (A - Y)
        and is handled directly in backward().

        Parameters
        ----------
        z    : np.ndarray - pre-activation
        name : str - 'relu', 'tanh', or 'sigmoid'

        Returns
        -------
        np.ndarray, same shape as z
        """
        if name == 'relu':
            return relu_derivative(z)
        elif name == 'tanh':
            return 1.0 - np.tanh(z) ** 2
        elif name == 'sigmoid':
            s = self._apply_activation(z, 'sigmoid')
            return s * (1.0 - s)
        else:
            raise ValueError(
                f"_activation_derivative called for '{name}' - "
                "softmax derivative is handled in backward()."
            )

    # ── Section 2.1 - Weight Initialisation ──────────────────────────────────

    def initialise_params(self):
        """Xavier (Glorot) initialisation for all layers.

        For layer l connecting n_in → n_out units:
            W_l ~ N(0, 1) * sqrt(1 / n_in)   shape (n_out, n_in)
            b_l = 0                            shape (n_out,)

        Store results in self.params_ under keys:
            'W1', 'b1', 'W2', 'b2', ..., 'WL', 'bL'
        where L = len(self.layer_sizes) - 1 (number of weight matrices).

        Hint:
            rng = np.random.default_rng(self.random_state)
            for l, (n_in, n_out) in enumerate(zip(self.layer_sizes[:-1],
                                                   self.layer_sizes[1:]), 1):
                self.params_[f'W{l}'] = rng.normal(0, 1, (n_out, n_in)) * np.sqrt(1 / n_in)
                self.params_[f'b{l}'] = np.zeros(n_out)
        """
        raise NotImplementedError("TODO: implement initialise_params")

    # ── Section 2.2 - Forward Pass ────────────────────────────────────────────

    def forward(self, X):
        """Forward pass through all layers.

        For layer l = 1, 2, …, L:
            Z_l = A_{l-1} @ W_l.T + b_l     where A_0 = X
            A_l = activation_l(Z_l)

        Parameters
        ----------
        X : np.ndarray shape (N, n_features)

        Returns
        -------
        A_L  : np.ndarray - output activations (probabilities)
        cache: dict - all intermediate values needed for backprop:
            'A0': X,
            'Z1': ..., 'A1': ...,
            'Z2': ..., 'A2': ...,
            ...,
            'ZL': ..., 'AL': ...
        Keys follow the pattern 'Z{l}' and 'A{l}' for l = 1..L.

        Hint: loop over l = 1..L, build cache['A0'] = X first, then
        compute Z and A for each layer using self.params_ and
        self._apply_activation.
        """
        raise NotImplementedError("TODO: implement forward")

    # ── Section 2.3 - Loss ───────────────────────────────────────────────────

    def compute_loss(self, A_out, y):
        """Compute the loss based on the output activation type.

        If output activation is 'softmax': categorical cross-entropy
            L = -1/N * Σ_i Σ_k  Y_ik * log(A_ik + ε)
            where Y is the one-hot encoding of y, ε = 1e-12

        If output activation is 'sigmoid': binary cross-entropy
            L = -1/N * Σ [ y*log(ŷ + ε) + (1-y)*log(1-ŷ + ε) ]

        Parameters
        ----------
        A_out : np.ndarray shape (N, K) or (N, 1)
        y     : np.ndarray shape (N,) - integer class labels (or binary)

        Returns
        -------
        float - scalar loss

        Hint for softmax:
            Y_onehot = np.eye(K)[y.astype(int)]   # one-hot  (N, K)
            loss = -np.mean(np.sum(Y_onehot * np.log(A_out + 1e-12), axis=1))

        Hint for sigmoid:
            y_col = y.reshape(-1, 1)
            loss = -np.mean(y_col * np.log(A_out + 1e-12)
                            + (1 - y_col) * np.log(1 - A_out + 1e-12))
        """
        raise NotImplementedError("TODO: implement compute_loss")

    # ── Section 2.4 - Backward Pass ──────────────────────────────────────────

    def backward(self, y, cache):
        """Backpropagation through all layers.

        Start from the output layer and propagate gradients backwards.

        Output layer gradient (dZ_L):
        ─────────────────────────────
        • softmax + categorical cross-entropy (elegant combined result):
              dZ_L = (A_L - Y_onehot) / N     shape (N, K)
        • sigmoid + binary cross-entropy:
              dZ_L = (A_L - y_col) / N         shape (N, 1)

        For each layer l = L, L-1, …, 1:
        ──────────────────────────────────
            dW_l = dZ_l.T @ A_{l-1}            shape (n_out, n_in)
            db_l = dZ_l.sum(axis=0)            shape (n_out,)
            if l > 1:
                dA_{l-1} = dZ_l @ W_l          shape (N, n_in)
                dZ_{l-1} = dA_{l-1} * activation_derivative(Z_{l-1})

        Parameters
        ----------
        y     : np.ndarray shape (N,) - true labels
        cache : dict - from forward()

        Returns
        -------
        grads : dict with keys 'dW1', 'db1', 'dW2', 'db2', …, 'dWL', 'dbL'

        Hint:
            L = len(self.layer_sizes) - 1
            N = cache['A0'].shape[0]
            K = self.layer_sizes[-1]

            # --- output layer ---
            if self.activations[-1] == 'softmax':
                Y_onehot = np.eye(K)[y.astype(int)]
                dZ = (cache[f'A{L}'] - Y_onehot) / N
            else:  # sigmoid
                dZ = (cache[f'A{L}'] - y.reshape(-1, 1)) / N

            grads = {}
            for l in range(L, 0, -1):
                grads[f'dW{l}'] = dZ.T @ cache[f'A{l-1}']
                grads[f'db{l}'] = dZ.sum(axis=0)
                if l > 1:
                    dA = dZ @ self.params_[f'W{l}']
                    dZ = dA * self._activation_derivative(cache[f'Z{l-1}'],
                                                          self.activations[l-2])
        """
        raise NotImplementedError("TODO: implement backward")

    # ── Section 2.5 - Parameter Update ───────────────────────────────────────

    def update_params(self, grads, lr):
        """Gradient descent update for all parameters (in-place).

            W_l ← W_l - lr * dW_l
            b_l ← b_l - lr * db_l

        Parameters
        ----------
        grads : dict - from backward()
        lr    : float - learning rate

        Hint: loop over l = 1..L and update self.params_['W{l}'] and
        self.params_['b{l}'] using grads['dW{l}'] and grads['db{l}'].
        """
        raise NotImplementedError("TODO: implement update_params")

    # ── Provided: Training Loop ───────────────────────────────────────────────

    def fit(self, X, y, lr=0.01, n_epochs=1000, verbose=True):
        """Train the network with full-batch gradient descent.

        Parameters
        ----------
        X        : np.ndarray shape (N, n_features)
        y        : np.ndarray shape (N,) - integer class labels
        lr       : float - learning rate
        n_epochs : int   - number of gradient descent steps
        verbose  : bool  - print loss every 200 epochs

        Returns
        -------
        self
        """
        X = np.array(X, dtype=float)
        y = np.array(y)
        self.loss_curve_ = []
        self.initialise_params()

        for epoch in range(n_epochs):
            A_out, cache = self.forward(X)
            loss = self.compute_loss(A_out, y)
            self.loss_curve_.append(loss)
            grads = self.backward(y, cache)
            self.update_params(grads, lr)

            if verbose and (epoch % 200 == 0 or epoch == n_epochs - 1):
                print(f"  Epoch {epoch:>5d}  loss={loss:.4f}")

        return self

    # ── Provided: Prediction ──────────────────────────────────────────────────

    def predict_proba(self, X):
        """Return output probabilities.

        Returns
        -------
        np.ndarray shape (N, K) for softmax, (N,) for sigmoid
        """
        X = np.array(X, dtype=float)
        A_out, _ = self.forward(X)
        if self.activations[-1] == 'sigmoid':
            return A_out.ravel()
        return A_out  # (N, K)

    def predict(self, X):
        """Predict class labels.

        Returns
        -------
        np.ndarray shape (N,) - integer class labels
        """
        proba = self.predict_proba(X)
        if proba.ndim == 1:
            return (proba >= 0.5).astype(int)
        return np.argmax(proba, axis=1)
