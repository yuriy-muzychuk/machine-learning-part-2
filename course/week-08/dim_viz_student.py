"""
dim_viz_student.py - t-SNE from Scratch (Student Version)
==========================================================
Week 8 Lab: Implement t-SNE for 2-D visualisation of high-dimensional data.
PCA (from week 7) and UMAP (via umap-learn) are used for comparison - no
implementation is required for those two.

Instructions
------------
Implement every method marked with
    raise NotImplementedError("TODO: ...")

The step-by-step comments inside each method describe exactly what to write.
Do NOT import anything beyond what is already imported.
Do NOT change method signatures or the __init__ constructor.

Dataset note
------------
The student's t-SNE runs in O(n^2) time. The notebook uses a 300-sample
subset of digits so that compute_P finishes in roughly 15-30 seconds.
Do not run on the full dataset (1 797 samples) without patience.
"""

import numpy as np


# =============================================================================
# Section 1 - Pairwise squared Euclidean distances
# =============================================================================

def pairwise_sq_distances(X):
    """Return the matrix of pairwise squared Euclidean distances.

    D[i, j] = ||x_i - x_j||^2

    Parameters
    ----------
    X : ndarray (n, d)

    Returns
    -------
    D : ndarray (n, n)
        Symmetric, non-negative, zero diagonal.

    Steps
    -----
    1. Compute the squared norm of every row:
           sum_sq = np.sum(X ** 2, axis=1)      # shape (n,)

    2. Use broadcasting to build the outer-sum of squared norms:
           sum_sq[:, None]   has shape (n, 1)
           sum_sq[None, :]   has shape (1, n)
           Their sum          has shape (n, n)

    3. Subtract twice the dot-product matrix (expansion identity):
           D = sum_sq[:, None] + sum_sq[None, :] - 2.0 * (X @ X.T)

    4. Zero the diagonal to remove floating-point self-distance noise:
           np.fill_diagonal(D, 0.0)

    5. Clip negative values caused by floating-point errors:
           D = np.maximum(D, 0.0)

    6. Return D.
    """
    raise NotImplementedError("TODO: implement pairwise_sq_distances")


# =============================================================================
# Section 2 - High-dimensional affinities (Gaussian kernel / perplexity)
# =============================================================================

def gaussian_affinities(d_sq_row, sigma):
    """Conditional probabilities p_{j|i} for one anchor point i.

    p_{j|i} = exp(-d_{ij}^2 / (2 sigma^2)) / sum_{k: d_{ik}>0} exp(-d_{ik}^2 / (2 sigma^2))

    The self-term (d = 0) is excluded before normalisation.

    Parameters
    ----------
    d_sq_row : ndarray (n,)  - row i of the squared-distance matrix;
                               d_sq_row[i] == 0 marks the anchor point
    sigma    : float         - bandwidth for this anchor

    Returns
    -------
    p_row : ndarray (n,)     - conditional probabilities; p_row[i] == 0

    Steps
    -----
    1. Compute unnormalised log-weights and shift for numerical stability:
           shifted = -d_sq_row / (2.0 * sigma ** 2)
           shifted -= shifted.max()        # subtract max to avoid overflow

    2. Exponentiate:
           exp_vals = np.exp(shifted)

    3. Zero out the self-term (the entry where d_sq_row == 0):
           exp_vals[d_sq_row == 0.0] = 0.0

    4. Sum the remaining weights:
           total = exp_vals.sum()

    5. Guard against the degenerate case (all zeros):
           if total == 0.0: return exp_vals

    6. Divide and return:
           return exp_vals / total
    """
    raise NotImplementedError("TODO: implement gaussian_affinities")


def _perplexity_of(p_row):
    """Shannon perplexity: 2^H  where H = -sum_j p_j log2(p_j).

    Helper used inside find_sigma - already implemented, do not change.
    """
    p = p_row[p_row > 0.0]
    H = -np.sum(p * np.log2(p))
    return 2.0 ** H


def find_sigma(d_sq_row, perplexity, tol=1e-5, max_iter=200):
    """Binary search for the bandwidth sigma_i that achieves the target perplexity.

    Perplexity is a smooth, strictly increasing function of sigma, so binary
    search on the interval [sigma_lo, sigma_hi] is guaranteed to converge.

    Parameters
    ----------
    d_sq_row   : ndarray (n,)   - row of the squared-distance matrix
    perplexity : float          - target perplexity (typical range: 5-50)
    tol        : float          - stop when |achieved - target| < tol
    max_iter   : int

    Returns
    -------
    sigma : float

    Steps
    -----
    1. Initialise the search bracket and starting point:
           sigma_lo = 1e-5
           sigma_hi = 1e5
           sigma    = 1.0

    2. Loop for max_iter iterations:
         a. Compute conditional probabilities for the current sigma:
                p_row = gaussian_affinities(d_sq_row, sigma)
         b. Compute the achieved perplexity:
                current_perp = _perplexity_of(p_row)
         c. Compute the error:
                diff = current_perp - perplexity
         d. If abs(diff) < tol: break  (close enough)
         e. Update the bracket (binary search rule):
                if diff > 0:           # perplexity too HIGH -> sigma is too large
                    sigma_hi = sigma
                else:                  # perplexity too LOW  -> sigma is too small
                    sigma_lo = sigma
         f. Move to the midpoint:
                sigma = (sigma_lo + sigma_hi) / 2.0

    3. Return sigma.
    """
    raise NotImplementedError("TODO: implement find_sigma")


def compute_P(X, perplexity=30.0):
    """Build the symmetrised joint probability matrix P for the full dataset.

    Algorithm
    ---------
    For each point i:
      1. call find_sigma(D[i], perplexity) to get sigma_i
      2. call gaussian_affinities(D[i], sigma_i) to get the conditional row

    Then symmetrise and normalise so the whole matrix sums to 1:
         p_{ij} = (p_{j|i} + p_{i|j}) / (2 * n)

    Parameters
    ----------
    X          : ndarray (n, d)
    perplexity : float

    Returns
    -------
    P : ndarray (n, n)
        Symmetric, sums to 1 (approximately), zero diagonal, all entries >= 0.

    Steps
    -----
    1.  n = X.shape[0]

    2.  Compute the full squared-distance matrix:
            D = pairwise_sq_distances(X)

    3.  Allocate a (n, n) matrix for conditional probabilities:
            P_cond = np.zeros((n, n))

    4.  Fill row by row (this loop is the slow O(n^2) step - ~15-30 s for n=300):
            for i in range(n):
                sigma_i    = find_sigma(D[i], perplexity)
                P_cond[i]  = gaussian_affinities(D[i], sigma_i)

    5.  Symmetrise and normalise (the 2*n factor ensures P sums to ~1):
            P = (P_cond + P_cond.T) / (2.0 * n)

    6.  Apply a numerical floor to avoid log(0) in the KL divergence:
            P = np.maximum(P, 1e-12)

    7.  Zero the diagonal (self-affinities play no role):
            np.fill_diagonal(P, 0.0)

    8.  Return P.
    """
    raise NotImplementedError("TODO: implement compute_P")


# =============================================================================
# Section 3 - Low-dimensional affinities (Student-t / Cauchy kernel)
# =============================================================================

def compute_Q(Y):
    """Joint probability matrix Q in the low-dimensional embedding space.

    Uses a Student-t kernel with one degree of freedom (Cauchy distribution):
        w_{ij}  = (1 + ||y_i - y_j||^2)^{-1}        (unnormalised weight)
        q_{ij}  = w_{ij} / sum_{k != l} w_{kl}       (normalised, excl. diagonal)

    The heavy tail of the t-distribution allows points that are moderately
    far apart in high-D to be placed much further apart in 2-D, which
    resolves the "crowding problem" inherent in a Gaussian low-dim kernel.

    Parameters
    ----------
    Y : ndarray (n, n_components)

    Returns
    -------
    Q : ndarray (n, n)
        Symmetric, sums to 1 (approximately), zero diagonal, all entries >= 1e-12.

    Steps
    -----
    1.  Compute the squared-distance matrix for the current embedding:
            D = pairwise_sq_distances(Y)

    2.  Compute unnormalised Student-t weights:
            W = 1.0 / (1.0 + D)

    3.  Zero the diagonal (exclude self-pairs from the normalisation):
            np.fill_diagonal(W, 0.0)

    4.  Normalise so that the whole matrix sums to 1:
            Q = W / W.sum()

    5.  Apply a numerical floor (same reason as in compute_P):
            Q = np.maximum(Q, 1e-12)

    6.  Return Q.
    """
    raise NotImplementedError("TODO: implement compute_Q")


# =============================================================================
# Section 4 - Gradient of the KL divergence
# =============================================================================

def tsne_gradient(P, Q, Y):
    """Gradient of KL(P || Q) with respect to the 2-D embedding Y.

    The analytic gradient for point i is:
        dC/dy_i = 4 * sum_j (p_{ij} - q_{ij}) * (y_i - y_j) * w_{ij}

    where w_{ij} = (1 + ||y_i - y_j||^2)^{-1}  is the Student-t weight.

    Parameters
    ----------
    P : ndarray (n, n)  - high-dim joint probabilities (from compute_P)
    Q : ndarray (n, n)  - low-dim  joint probabilities (from compute_Q)
    Y : ndarray (n, n_components)

    Returns
    -------
    grad : ndarray (n, n_components)

    Steps
    -----
    1.  n = Y.shape[0]

    2.  Compute squared distances and Student-t weights for the current Y:
            D = pairwise_sq_distances(Y)
            W = 1.0 / (1.0 + D)          # shape (n, n)
            np.fill_diagonal(W, 0.0)

    3.  Build the combined scalar multiplier matrix:
            PQ_diff = (P - Q) * W         # shape (n, n), element-wise product

    4.  Compute the gradient. Two equivalent options - pick one:

        Option A (vectorised with einsum - fast):
            # Y[:, None, :] - Y[None, :, :] gives pairwise differences
            # shape (n, n, n_components)
            Y_diff = Y[:, None, :] - Y[None, :, :]
            grad   = 4.0 * np.einsum("ij,ijk->ik", PQ_diff, Y_diff)

        Option B (explicit loop - easier to read):
            grad = np.zeros_like(Y)
            for i in range(n):
                # diff[j] = y_i - y_j, shape (n, n_components)
                diff    = Y[i] - Y                      # (n, n_components)
                # weight each difference by (p_ij - q_ij) * w_ij
                weights = PQ_diff[i]                    # (n,)
                grad[i] = 4.0 * np.sum(weights[:, None] * diff, axis=0)

    5.  Return grad.
    """
    raise NotImplementedError("TODO: implement tsne_gradient")


# =============================================================================
# Section 5 - TSNE class
# =============================================================================

class TSNE:
    """Simplified t-SNE for 2-D (or n_components-D) visualisation.

    Covers the core algorithm:
      - Gaussian high-dim affinities with perplexity-calibrated bandwidth
      - Student-t low-dim affinities (Cauchy kernel)
      - Gradient descent with momentum

    Intentionally omits production tricks (early exaggeration, Barnes-Hut
    approximation, adaptive learning rate) to keep the code transparent.

    Parameters
    ----------
    n_components  : int    - embedding dimension (almost always 2)
    perplexity    : float  - effective neighbourhood size; typical range 5-50
    n_iter        : int    - number of gradient-descent steps
    learning_rate : float  - step size (eta)
    momentum      : float  - momentum coefficient (mu); 0 disables momentum
    random_state  : int
    """

    def __init__(
        self,
        n_components=2,
        perplexity=30.0,
        n_iter=500,
        learning_rate=200.0,
        momentum=0.8,
        random_state=42,
    ):
        self.n_components  = n_components
        self.perplexity    = perplexity
        self.n_iter        = n_iter
        self.learning_rate = learning_rate
        self.momentum      = momentum
        self.random_state  = random_state

        self.embedding_    = None   # set by fit_transform
        self.kl_curve_     = []     # KL divergence recorded at each iteration

    # -------------------------------------------------------------------------

    def fit_transform(self, X):
        """Compute and return the t-SNE embedding of X.

        Parameters
        ----------
        X : ndarray (n, d) - high-dimensional input; should be pre-scaled
                             (e.g. via StandardScaler)

        Returns
        -------
        Y : ndarray (n, n_components) - the 2-D embedding

        Steps
        -----
        1.  Create a random-number generator for reproducibility:
                rng = np.random.RandomState(self.random_state)
                n   = X.shape[0]

        2.  Compute high-dim affinities - the expensive step, done once:
                P = compute_P(X, self.perplexity)

        3.  Initialise the embedding with small random values (keeping points
            close together avoids large initial gradients):
                Y = rng.randn(n, self.n_components) * 1e-4

        4.  Initialise the velocity vector (same shape as Y, all zeros):
                velocity = np.zeros_like(Y)

        5.  Reset the loss log:
                self.kl_curve_ = []

        6.  Gradient-descent loop - for t in range(self.n_iter):

              a. Compute low-dim affinities for current Y:
                     Q = compute_Q(Y)

              b. Compute the gradient:
                     grad = tsne_gradient(P, Q, Y)

              c. Momentum update (replaces the plain SGD step):
                     velocity = self.momentum * velocity - self.learning_rate * grad
                     Y        = Y + velocity

              d. Record the KL divergence for monitoring convergence.
                 Use masked indexing to avoid log(0) warnings:
                     mask = P > 0
                     kl   = np.sum(P[mask] * (np.log(P[mask]) - np.log(Q[mask])))
                     self.kl_curve_.append(kl)

        7.  Store and return the final embedding:
                self.embedding_ = Y
                return Y
        """
        raise NotImplementedError("TODO: implement fit_transform")
