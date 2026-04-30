"""
density_estimation_student.py - KDE and GMM from Scratch (Student Version)
===========================================================================
Week 9 Lab: Density Estimation

Your task
---------
Implement the four methods marked with
Everything else - including the core algorithmic logic - is already provided
so that you can run and explore the notebook while working on your parts.

Methods YOU implement:
    KDE._log_gaussian_kernel   - formula for the log of a Gaussian kernel
    KDE.sample                 - draw samples from the estimated density
    GMM.score_samples          - log-likelihood of each point under the mixture
    GMM.predict                - hard cluster assignment via argmax

Methods PROVIDED for you:
    KDE.fit                    - store training data
    KDE.score_samples          - full log-density computation (calls your kernel)
    GMM._log_gaussian          - log N(X | mu, Sigma) helper
    GMM.fit                    - full EM loop (calls _e_step and _m_step)
    GMM._e_step                - E-step: compute responsibilities
    GMM._m_step                - M-step: update parameters
    GMM.sample                 - generate samples from the fitted mixture

Do NOT import anything beyond what is already imported.
Do NOT change method signatures or __init__ constructors.
"""

import numpy as np
from scipy.special import logsumexp


# =============================================================================
# Section 1 - Kernel Density Estimation (KDE)
# =============================================================================

class KDE:
    """Gaussian Kernel Density Estimator.

    Estimates p(x) given training data as a sum of Gaussian kernels:

        p_hat(x) = (1/n) * sum_{i=1}^{n} (1/h^d) * K( (x - x_i) / h )

    Parameters
    ----------
    bandwidth : float  - smoothing bandwidth h (must be > 0)
    """

    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth
        self.X_train_  = None

    # ------------------------------------------------------------------
    # Provided
    # ------------------------------------------------------------------

    def fit(self, X):
        """Store the training data. KDE has no parameters to optimise."""
        self.X_train_ = np.asarray(X, dtype=float)
        return self

    def score_samples(self, X_query):
        """Log-density estimate at each query point.

        Calls your _log_gaussian_kernel internally - implement that first.

        Parameters
        ----------
        X_query : ndarray (m, d)

        Returns
        -------
        log_density : ndarray (m,)
        """
        X_query = np.asarray(X_query, dtype=float)
        n, d    = self.X_train_.shape

        # diff[i, j, k] = (X_query[i,k] - X_train[j,k]) / h
        diff  = (X_query[:, None, :] - self.X_train_[None, :, :]) / self.bandwidth

        # sum log-kernel over feature dimensions
        log_k = np.sum(self._log_gaussian_kernel(diff), axis=-1)   # (m, n)

        # log-sum-exp over training points minus normalisation constant
        return logsumexp(log_k, axis=1) - np.log(n) - d * np.log(self.bandwidth)

    # ------------------------------------------------------------------
    # TODO
    # ------------------------------------------------------------------

    def _log_gaussian_kernel(self, u):
        """Log of the standard Gaussian kernel evaluated at u.

        The per-dimension Gaussian kernel is:
            K(u) = (1 / sqrt(2*pi)) * exp(-0.5 * u^2)

        Its log is:
            log K(u) = -0.5 * log(2*pi) - 0.5 * u^2

        Parameters
        ----------
        u : ndarray  - standardised value (x - x_i) / h, any shape

        Returns
        -------
        log_k : ndarray  - same shape as u

        Steps
        -----
        1. Compute -0.5 * np.log(2.0 * np.pi)
        2. Compute -0.5 * u ** 2
        3. Return their sum.
        """
        raise NotImplementedError("TODO: implement _log_gaussian_kernel")

    def sample(self, n_samples, random_state=None):
        """Draw samples from the estimated density (kernel smoothing bootstrap).

        Algorithm:
          1. Pick n_samples training points uniformly at random (with replacement).
          2. Add isotropic Gaussian noise ~ N(0, h^2 * I) to each.

        Parameters
        ----------
        n_samples    : int
        random_state : int or None

        Returns
        -------
        samples : ndarray (n_samples, d)

        Steps
        -----
        1. rng = np.random.RandomState(random_state)
        2. Read n, d from self.X_train_.shape.
        3. idx = rng.randint(0, n, size=n_samples)
        4. centers = self.X_train_[idx]
        5. noise = rng.randn(n_samples, d) * self.bandwidth
        6. Return centers + noise.
        """
        raise NotImplementedError("TODO: implement sample")


# =============================================================================
# Section 2 - Gaussian Mixture Model (GMM)
# =============================================================================

class GMM:
    """Gaussian Mixture Model fitted by the EM algorithm.

    Model:
        p(x) = sum_{k=1}^{K}  pi_k * N(x; mu_k, Sigma_k)

    Parameters
    ----------
    n_components : int   - number of mixture components K
    n_iter       : int   - maximum EM iterations
    tol          : float - stop when |delta log-likelihood| < tol
    random_state : int
    """

    def __init__(self, n_components=3, n_iter=200, tol=1e-6, random_state=42):
        self.n_components = n_components
        self.n_iter       = n_iter
        self.tol          = tol
        self.random_state = random_state

        self.weights_     = None   # pi_k,    shape (K,)
        self.means_       = None   # mu_k,    shape (K, d)
        self.covariances_ = None   # Sigma_k, shape (K, d, d)
        self.log_likelihood_curve_ = []

    # ------------------------------------------------------------------
    # Provided
    # ------------------------------------------------------------------

    def _log_gaussian(self, X, mu, Sigma):
        """Log N(X | mu, Sigma) for every row of X (Cholesky-based)."""
        d    = X.shape[1]
        diff = X - mu
        try:
            L       = np.linalg.cholesky(Sigma)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            alpha   = np.linalg.solve(L, diff.T).T
            maha    = np.sum(alpha ** 2, axis=1)
        except np.linalg.LinAlgError:
            sign, log_det = np.linalg.slogdet(Sigma)
            inv_S = np.linalg.pinv(Sigma)
            maha  = np.sum(diff @ inv_S * diff, axis=1)
        return -0.5 * (d * np.log(2.0 * np.pi) + log_det + maha)

    def fit(self, X):
        """Fit GMM via the EM algorithm.

        Initialises means by randomly sampling K data points, covariances
        as identity matrices, mixing weights as uniform 1/K, then iterates
        _e_step / _m_step until convergence.
        """
        X    = np.asarray(X, dtype=float)
        n, d = X.shape
        K    = self.n_components
        rng  = np.random.RandomState(self.random_state)

        idx               = rng.choice(n, K, replace=False)
        self.means_       = X[idx].copy()
        self.covariances_ = np.array([np.eye(d) for _ in range(K)])
        self.weights_     = np.ones(K) / K

        self.log_likelihood_curve_ = []
        prev_ll = -np.inf

        for _ in range(self.n_iter):
            R  = self._e_step(X)
            self._m_step(X, R)
            ll = np.sum(self.score_samples(X))
            self.log_likelihood_curve_.append(ll)
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        return self

    def _e_step(self, X):
        """Compute soft assignments (responsibilities).

        r_{ik} = pi_k * N(x_i; mu_k, Sigma_k)
                 / sum_j pi_j * N(x_i; mu_j, Sigma_j)

        Returns R : ndarray (n, K) with rows summing to 1.
        """
        K        = self.n_components
        n        = X.shape[0]
        log_resp = np.zeros((n, K))

        for k in range(K):
            log_resp[:, k] = (np.log(self.weights_[k] + 1e-300)
                              + self._log_gaussian(X, self.means_[k], self.covariances_[k]))

        log_resp -= logsumexp(log_resp, axis=1, keepdims=True)
        return np.exp(log_resp)

    def _m_step(self, X, R):
        """Re-estimate parameters from responsibilities.

        N_k = sum_i R[i,k]
        pi_k = N_k / n
        mu_k = (1/N_k) sum_i R[i,k] x_i
        Sigma_k = (1/N_k) sum_i R[i,k] (x_i - mu_k)(x_i - mu_k)^T + 1e-6 I
        """
        n, d = X.shape
        K    = self.n_components
        Nk   = R.sum(axis=0)

        self.weights_ = Nk / n
        self.means_   = (R.T @ X) / Nk[:, None]

        for k in range(K):
            diff                 = X - self.means_[k]
            weighted             = R[:, k:k+1] * diff
            self.covariances_[k] = (weighted.T @ diff) / Nk[k]
            self.covariances_[k] += 1e-6 * np.eye(d)

    def sample(self, n_samples):
        """Generate samples from the fitted mixture.

        1. Sample a component index k ~ Categorical(pi).
        2. Sample x ~ N(mu_k, Sigma_k).

        Returns samples : ndarray (n_samples, d)
        """
        rng           = np.random.RandomState(self.random_state)
        K, d          = self.n_components, self.means_.shape[1]
        component_idx = rng.choice(K, size=n_samples, p=self.weights_)
        samples       = np.zeros((n_samples, d))

        for i, k in enumerate(component_idx):
            samples[i] = rng.multivariate_normal(self.means_[k], self.covariances_[k])

        return samples

    # ------------------------------------------------------------------
    # TODO
    # ------------------------------------------------------------------

    def score_samples(self, X):
        """Log-likelihood of each data point under the mixture.

        log p(x_i) = log sum_{k} pi_k * N(x_i; mu_k, Sigma_k)

        Hint: the pattern is the same as _e_step steps 1-2, but here you
        do NOT normalise - just apply logsumexp to marginalise over k.

        Parameters
        ----------
        X : ndarray (n, d)

        Returns
        -------
        log_probs : ndarray (n,)

        Steps
        -----
        1. X = np.asarray(X, dtype=float)
        2. K = self.n_components
        3. log_comp = np.zeros((X.shape[0], K))
        4. For k in range(K):
               log_comp[:, k] = np.log(self.weights_[k] + 1e-300) \
                                + self._log_gaussian(X, self.means_[k], self.covariances_[k])
        5. return logsumexp(log_comp, axis=1)
        """
        raise NotImplementedError("TODO: implement score_samples")

    def predict(self, X):
        """Hard cluster assignment: argmax_k p(z=k | x_i).

        Parameters
        ----------
        X : ndarray (n, d)

        Returns
        -------
        labels : ndarray (n,) of int in [0, K-1]

        Steps
        -----
        1. X = np.asarray(X, dtype=float)
        2. K = self.n_components
        3. Build the unnormalised log-responsibility matrix (same as the
           first loop in _e_step, without the normalisation line):
               log_resp = np.zeros((X.shape[0], K))
               for k in range(K):
                   log_resp[:, k] = np.log(self.weights_[k] + 1e-300) \
                                    + self._log_gaussian(X, self.means_[k], self.covariances_[k])
        4. return np.argmax(log_resp, axis=1)
        """
        raise NotImplementedError("TODO: implement predict")
