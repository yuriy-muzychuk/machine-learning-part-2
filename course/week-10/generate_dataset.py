"""
generate_dataset.py - Synthetic Anomaly Detection Dataset
==========================================================
Run this script once before starting Lab 10 Part 6.
Produces  anomaly_dataset.csv  in the current working directory.

Dataset design  (~5 000 rows, ~10 % anomalies)
-----------------------------------------------
Normal data  (label = 0)
  Cluster A : tight Gaussian  at ( 0,  0),  n = 2 200
  Cluster B : wider Gaussian  at ( 5,  4),  n = 2 000
  Cluster C : small Gaussian  at (-4,  5),  n =   300

Anomalies  (label = 1)
  Type 1 - distant scatter   : uniform in [-10, 10]^2,
            Mahalanobis distance > 4 from every normal centre,  n ≈ 250
  Type 2 - compact cluster   : tight Gaussian at (8, -3),
            well separated from all normal clusters,            n = 150
  Type 3 - near-boundary ring: ring around Cluster A at
            Mahalanobis radius 2.8–3.8,                         n ≈ 100

Total ≈ 5 000 rows.  Columns: x1, x2, label.
"""

import numpy as np
import pandas as pd

SEED = 42
rng  = np.random.RandomState(SEED)

# ── Normal clusters ───────────────────────────────────────────────────────────

cov_A = np.array([[1.0, 0.4], [0.4, 0.8]])
X_A   = rng.multivariate_normal([0.0, 0.0], cov_A, size=2200)

cov_B = np.array([[2.0, -0.5], [-0.5, 1.5]])
X_B   = rng.multivariate_normal([5.0, 4.0], cov_B, size=2000)

cov_C = np.array([[0.6, 0.0], [0.0, 0.6]])
X_C   = rng.multivariate_normal([-4.0, 5.0], cov_C, size=300)

X_normal = np.vstack([X_A, X_B, X_C])
y_normal = np.zeros(len(X_normal), dtype=int)

# ── Helper: minimum Mahalanobis distance to any cluster centre ────────────────

def min_mahal(pts, centres):
    """Return the minimum Mahalanobis distance from each point to any centre.

    Parameters
    ----------
    pts     : ndarray (m, 2)
    centres : list of (mu, cov_inv) tuples

    Returns
    -------
    ndarray (m,)
    """
    dists = []
    for mu, cov_inv in centres:
        diff = pts - np.asarray(mu)
        d    = np.sqrt(np.einsum("ij,jk,ik->i", diff, cov_inv, diff))
        dists.append(d)
    return np.min(np.stack(dists, axis=1), axis=1)


centre_list = [
    ([0.0,  0.0], np.linalg.inv(cov_A)),
    ([5.0,  4.0], np.linalg.inv(cov_B)),
    ([-4.0, 5.0], np.linalg.inv(cov_C)),
]

# ── Anomaly type 1 - distant uniform scatter ──────────────────────────────────

candidates = rng.uniform(-10, 10, size=(5000, 2))
maha_cands = min_mahal(candidates, centre_list)
X_type1    = candidates[maha_cands > 4.0][:250]

# ── Anomaly type 2 - compact cluster at (8, -3) ───────────────────────────────

X_type2 = rng.multivariate_normal(
    [8.0, -3.0], [[0.3, 0.0], [0.0, 0.3]], size=150
)

# ── Anomaly type 3 - near-boundary ring around Cluster A ─────────────────────

angles = rng.uniform(0.0, 2.0 * np.pi, size=500)
radii  = rng.uniform(2.8, 3.8, size=500)
ring   = np.c_[radii * np.cos(angles), radii * np.sin(angles)]

cov_A_inv  = np.linalg.inv(cov_A)
diff_ring  = ring - np.array([0.0, 0.0])
maha_ring  = np.sqrt(np.einsum("ij,jk,ik->i", diff_ring, cov_A_inv, diff_ring))
X_type3    = ring[maha_ring > 3.0][:100]

# ── Combine, shuffle, and save ────────────────────────────────────────────────

X_anomaly = np.vstack([X_type1, X_type2, X_type3])
y_anomaly = np.ones(len(X_anomaly), dtype=int)

X_all = np.vstack([X_normal, X_anomaly])
y_all = np.concatenate([y_normal, y_anomaly])

shuffle_idx = rng.permutation(len(y_all))
X_all = X_all[shuffle_idx]
y_all = y_all[shuffle_idx]

df = pd.DataFrame({"x1": X_all[:, 0], "x2": X_all[:, 1], "label": y_all})
df.to_csv("anomaly_dataset.csv", index=False)

print("Saved anomaly_dataset.csv")
print(f"  Total rows : {len(df)}")
print(f"  Normal     : {(y_all == 0).sum()}  ({(y_all == 0).mean() * 100:.1f} %)")
print(f"  Anomaly    : {(y_all == 1).sum()}  ({(y_all == 1).mean() * 100:.1f} %)")
print()
print("Anomaly breakdown:")
print(f"  Type 1 (distant scatter) : {len(X_type1)}")
print(f"  Type 2 (compact cluster) : {len(X_type2)}")
print(f"  Type 3 (near-boundary)   : {len(X_type3)}")
