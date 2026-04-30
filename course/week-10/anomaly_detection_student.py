"""
anomaly_detection_student.py — Anomaly Detection (Student Version)
===================================================================
Week 10 Lab: Anomaly Detection

Classes defined here
--------------------
IsolationTree    — a single random binary isolation tree (from scratch)
IsolationForest  — ensemble of IsolationTrees             (from scratch)
OneClassSVM      — sklearn OCSVM with manual kernel eval  (wrapper)

Methods YOU implement  (6 TODOs)
---------------------------------
IsolationTree.fit             — recursive random binary split
IsolationTree.path_length     — path length to isolate one point
IsolationForest.score_samples — mean path length -> anomaly score in (0, 1)
IsolationForest.predict       — flag top-contamination fraction as anomalies
OneClassSVM._rbf_kernel       — RBF kernel matrix K[i,j] = exp(-g ||xi-yj||^2)
OneClassSVM.score_samples     — decision function from stored support vectors

Do NOT import anything beyond what is already imported.
Do NOT change method signatures or __init__ constructors.
"""

import numpy as np
from sklearn.svm import OneClassSVM as _SklearnOCSVM


# =============================================================================
# Helper — expected path length normalisation
# =============================================================================

def _c(n):
    """Expected path length of an unsuccessful search in a BST with n nodes.

    This term corrects for the fact that a leaf node may still contain
    more than one point when the tree ran out of depth.

    Formula (Liu et al., 2008):
        c(n) = 2 * H(n-1) - 2*(n-1)/n   for n >= 3
        c(2) = 1
        c(1) = 0

    where H(k) = ln(k) + 0.5772156649  (Euler-Mascheroni constant).

    Parameters
    ----------
    n : int — number of samples at the node

    Returns
    -------
    float
    """
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    H = np.log(n - 1) + 0.5772156649
    return 2.0 * H - 2.0 * (n - 1) / n


# =============================================================================
# Section 1 — Isolation Tree
# =============================================================================

class IsolationTree:
    """A single random binary tree that isolates data points by splitting.

    Building the tree
    -----------------
    At each node a random feature and a random split value are chosen.
    Splitting stops when the node has <= 1 point or max_depth is reached.

    Scoring a point
    ---------------
    path_length(x) returns the depth at which x is isolated.
    Anomalous points are isolated quickly (short path); normal points
    require many splits (long path) because they live in dense regions.

    Parameters
    ----------
    max_depth : int — maximum tree depth
    """

    def __init__(self, max_depth):
        self.max_depth = max_depth

        # Internal node attributes (set by fit)
        self.split_feature = None   # int: column index used for the split
        self.split_value   = None   # float: threshold value
        self.left          = None   # IsolationTree: x[feature] <  split_value
        self.right         = None   # IsolationTree: x[feature] >= split_value

        # Leaf node attributes (set by fit)
        self.is_leaf = False
        self.size    = None         # number of training points at this leaf

    # ------------------------------------------------------------------
    # TODO 1
    # ------------------------------------------------------------------

    def fit(self, X, depth=0):
        """Recursively build the isolation tree on dataset X.

        Parameters
        ----------
        X     : ndarray (n, d) — data points reaching this node
        depth : int            — current depth (root starts at 0)

        Returns
        -------
        self

        Steps
        -----
        1. If len(X) <= 1 OR depth >= self.max_depth, mark as a leaf:
               self.is_leaf = True
               self.size    = len(X)
               return self

        2. Pick a random split feature index in [0, d-1]:
               self.split_feature = np.random.randint(0, X.shape[1])

        3. Read the column and its range:
               col     = X[:, self.split_feature]
               lo, hi  = col.min(), col.max()

        4. If lo == hi (all values identical), mark as leaf and return.

        5. Sample the split value uniformly between lo and hi:
               self.split_value = np.random.uniform(lo, hi)

        6. Partition the data:
               mask    = col < self.split_value
               X_left  = X[mask]
               X_right = X[~mask]

        7. Recurse on each child:
               self.left  = IsolationTree(self.max_depth).fit(X_left,  depth + 1)
               self.right = IsolationTree(self.max_depth).fit(X_right, depth + 1)

        8. Return self.
        """
        raise NotImplementedError("TODO 1: implement IsolationTree.fit")

    # ------------------------------------------------------------------
    # TODO 2
    # ------------------------------------------------------------------

    def path_length(self, x, depth=0):
        """Return the effective path length needed to isolate point x.

        At a leaf the tree ran out of depth or data; the correction term
        _c(self.size) accounts for the additional splits that would have
        occurred if the tree were grown fully.

        Parameters
        ----------
        x     : ndarray (d,) — a single data point
        depth : int          — current depth (call from root with depth=0)

        Returns
        -------
        float — effective isolation depth

        Steps
        -----
        1. If self.is_leaf:
               return depth + _c(self.size)

        2. Route x to the correct child:
               if x[self.split_feature] < self.split_value:
                   return self.left.path_length(x, depth + 1)
               else:
                   return self.right.path_length(x, depth + 1)
        """
        raise NotImplementedError("TODO 2: implement IsolationTree.path_length")


# =============================================================================
# Section 2 — Isolation Forest
# =============================================================================

class IsolationForest:
    """Ensemble of IsolationTrees for unsupervised anomaly detection.

    Anomaly score (Liu et al., 2008):
        s(x) = 2 ^ { -E[h(x)] / c(max_samples) }

    where E[h(x)] is the mean path length over all trees and c(n) is the
    expected BST path length from _c().

    Interpretation:
        s(x) close to 1   — likely anomaly   (isolated quickly by all trees)
        s(x) close to 0.5 — likely normal    (hard to isolate)
        s(x) close to 0   — never anomaly    (very deep paths, denser than normal)

    Parameters
    ----------
    n_estimators  : int   — number of trees in the ensemble
    max_samples   : int   — subsample size per tree (None -> min(256, n))
    contamination : float — expected fraction of anomalies, used in predict()
    random_state  : int
    """

    def __init__(self, n_estimators=100, max_samples=None,
                 contamination=0.1, random_state=42):
        self.n_estimators  = n_estimators
        self.max_samples   = max_samples
        self.contamination = contamination
        self.random_state  = random_state

        self.trees_       = []
        self.max_samples_ = None   # resolved in fit()

    # ------------------------------------------------------------------
    # Provided
    # ------------------------------------------------------------------

    def fit(self, X):
        """Build the forest on training data X.

        Each tree receives an independent random subsample of max_samples rows.
        Tree depth is capped at ceil(log2(max_samples)) following Liu et al.
        The global numpy random state is seeded for reproducibility.
        """
        np.random.seed(self.random_state)
        rng = np.random.RandomState(self.random_state)
        n   = len(X)

        self.max_samples_ = (
            min(256, n) if self.max_samples is None else min(self.max_samples, n)
        )
        max_depth = int(np.ceil(np.log2(max(self.max_samples_, 2))))

        self.trees_ = []
        for _ in range(self.n_estimators):
            idx   = rng.choice(n, size=self.max_samples_, replace=False)
            tree  = IsolationTree(max_depth=max_depth)
            tree.fit(X[idx])
            self.trees_.append(tree)

        return self

    # ------------------------------------------------------------------
    # TODO 3
    # ------------------------------------------------------------------

    def score_samples(self, X):
        """Compute the anomaly score for each point in X.

        Higher scores indicate more anomalous points.

        Parameters
        ----------
        X : ndarray (n, d)

        Returns
        -------
        scores : ndarray (n,) — anomaly scores in (0, 1)

        Steps
        -----
        For each point x in X:
          1. Collect path lengths from every tree:
                 h_vals = [tree.path_length(x) for tree in self.trees_]

          2. Average them:
                 mean_h = np.mean(h_vals)

          3. Compute the anomaly score:
                 score = 2.0 ** (-mean_h / _c(self.max_samples_))

        Return an array of scores for all n points.
        """
        raise NotImplementedError("TODO 3: implement IsolationForest.score_samples")

    # ------------------------------------------------------------------
    # TODO 4
    # ------------------------------------------------------------------

    def predict(self, X):
        """Classify each point as normal (0) or anomaly (1).

        The threshold is chosen so that exactly the top `contamination`
        fraction of points receive label 1.

        Parameters
        ----------
        X : ndarray (n, d)

        Returns
        -------
        labels : ndarray (n,) — 0 = normal, 1 = anomaly

        Steps
        -----
        1. scores    = self.score_samples(X)
        2. threshold = np.percentile(scores, 100 * (1 - self.contamination))
        3. return (scores >= threshold).astype(int)
        """
        raise NotImplementedError("TODO 4: implement IsolationForest.predict")


# =============================================================================
# Section 3 — One-Class SVM  (sklearn wrapper)
# =============================================================================

class OneClassSVM:
    """One-Class SVM anomaly detector with manual RBF kernel evaluation.

    sklearn's OneClassSVM is used for the optimisation (QP solver).
    After fitting, the decision function is re-implemented here manually
    using the stored support vectors and dual coefficients.

    This makes the kernel trick concrete: to score a new point x, we only
    need to evaluate K(x, sv_i) for each support vector sv_i — the full
    training set is not required at prediction time.

    Decision boundary:
        f(x) = sum_i alpha_i * K(sv_i, x) - rho

    Points with f(x) >= 0 are classified as normal; f(x) < 0 as anomalies.

    Parameters
    ----------
    nu    : float — upper bound on training errors and lower bound on the
                    fraction of support vectors; controls the boundary tightness
    gamma : float — RBF kernel bandwidth (larger gamma = tighter kernel)
    """

    def __init__(self, nu=0.1, gamma=0.5):
        self.nu    = nu
        self.gamma = gamma

        self._svm             = None
        self._support_vectors = None   # ndarray (n_sv, d)
        self._dual_coef       = None   # ndarray (n_sv,)
        self._rho             = None   # float — decision threshold

    # ------------------------------------------------------------------
    # Provided
    # ------------------------------------------------------------------

    def fit(self, X):
        """Fit the One-Class SVM and cache support vector information."""
        self._svm = _SklearnOCSVM(kernel="rbf", nu=self.nu, gamma=self.gamma)
        self._svm.fit(X)

        self._support_vectors = self._svm.support_vectors_       # (n_sv, d)
        self._dual_coef       = self._svm.dual_coef_.ravel()     # (n_sv,)
        # sklearn stores intercept_ as -rho; we store rho directly
        self._rho             = -float(self._svm.intercept_[0])
        return self

    def predict(self, X):
        """Classify each point: 0 = normal, 1 = anomaly."""
        return (self.score_samples(X) < 0).astype(int)

    # ------------------------------------------------------------------
    # TODO 5
    # ------------------------------------------------------------------

    def _rbf_kernel(self, X, Y):
        """Compute the RBF kernel matrix between rows of X and rows of Y.

        K[i, j] = exp( -gamma * ||x_i - y_j||^2 )

        Parameters
        ----------
        X : ndarray (m, d)
        Y : ndarray (n, d)

        Returns
        -------
        K : ndarray (m, n)

        Steps
        -----
        1. Compute squared norms:
               sq_X = np.sum(X ** 2, axis=1, keepdims=True)   # (m, 1)
               sq_Y = np.sum(Y ** 2, axis=1, keepdims=True)   # (n, 1)

        2. Expand to pairwise squared distances:
               D_sq = sq_X + sq_Y.T - 2.0 * (X @ Y.T)         # (m, n)
               D_sq = np.maximum(D_sq, 0.0)                    # clip fp noise

        3. Apply the RBF formula:
               return np.exp(-self.gamma * D_sq)
        """
        raise NotImplementedError("TODO 5: implement OneClassSVM._rbf_kernel")

    # ------------------------------------------------------------------
    # TODO 6
    # ------------------------------------------------------------------

    def score_samples(self, X):
        """Compute the SVM decision function value for each point in X.

        Positive values indicate normal points; negative values indicate
        anomalies (consistent with sklearn's decision_function convention).

        Parameters
        ----------
        X : ndarray (n, d)

        Returns
        -------
        scores : ndarray (n,) — signed distance to the decision boundary

        Steps
        -----
        1. Compute the kernel matrix between query points and support vectors:
               K = self._rbf_kernel(X, self._support_vectors)   # (n, n_sv)

        2. Weighted sum over support vectors:
               scores = K @ self._dual_coef                     # (n,)

        3. Subtract the learned threshold:
               scores = scores - self._rho

        4. Return scores.
        """
        raise NotImplementedError("TODO 6: implement OneClassSVM.score_samples")
