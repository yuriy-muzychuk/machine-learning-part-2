"""
E-Commerce Customer Behavior Dataset Generator
===============================================
Generates a realistic dataset for multi-class customer segmentation.

Target variable: customer_segment
    - "Churned"  : lost customers (no recent activity, low engagement)
    - "At-Risk"  : declining engagement, may churn soon
    - "Loyal"    : regular, satisfied customers
    - "VIP"      : high-value, highly engaged customers

Binary collapse: ["Churned", "At-Risk"] -> 0  /  ["Loyal", "VIP"] -> 1

Features
--------
Numerical:
    age, total_spent, num_orders, avg_order_value, days_since_last_order,
    avg_session_duration_min, avg_pages_per_session, discount_usage_rate,
    num_support_tickets, loyalty_points

Categorical:
    region, device_type, payment_method, acquisition_channel, membership_tier

Corruption (~7% of data):
    - Random nulls across all columns
    - Impossible values: negative age, negative spent, days_since < 0
    - Type errors: numeric field filled with a string sentinel ("N/A", "error")

Usage
-----
    python generate_dataset.py                  # saves ecommerce_customers.csv
    python generate_dataset.py --seed 99        # custom seed
    python generate_dataset.py --n 2000         # custom size
"""

import argparse
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Segment profiles: (mean, std) or probability weights for each feature
# ---------------------------------------------------------------------------
SEGMENT_PROFILES = {
    "Churned": {
        "age":                      (38, 12),
        "total_spent":              (180,  80),
        "num_orders":               (3,    2),
        "avg_order_value":          (55,   20),
        "days_since_last_order":    (280,  60),
        "avg_session_duration_min": (4,    3),
        "avg_pages_per_session":    (3,    2),
        "discount_usage_rate":      (0.6,  0.2),
        "num_support_tickets":      (3,    2),
        "loyalty_points":           (120,  80),
        "weight": 0.22,
    },
    "At-Risk": {
        "age":                      (34, 10),
        "total_spent":              (420,  150),
        "num_orders":               (8,    4),
        "avg_order_value":          (65,   25),
        "days_since_last_order":    (90,   30),
        "avg_session_duration_min": (9,    4),
        "avg_pages_per_session":    (6,    3),
        "discount_usage_rate":      (0.45, 0.2),
        "num_support_tickets":      (2,    1.5),
        "loyalty_points":           (380,  150),
        "weight": 0.28,
    },
    "Loyal": {
        "age":                      (31, 9),
        "total_spent":              (950,  300),
        "num_orders":               (22,   8),
        "avg_order_value":          (85,   30),
        "days_since_last_order":    (18,   12),
        "avg_session_duration_min": (16,   5),
        "avg_pages_per_session":    (11,   4),
        "discount_usage_rate":      (0.25, 0.15),
        "num_support_tickets":      (1,    1),
        "loyalty_points":           (1100, 400),
        "weight": 0.35,
    },
    "VIP": {
        "age":                      (40, 11),
        "total_spent":              (3500, 1200),
        "num_orders":               (60,   20),
        "avg_order_value":          (145,  50),
        "days_since_last_order":    (7,    5),
        "avg_session_duration_min": (28,   8),
        "avg_pages_per_session":    (18,   5),
        "discount_usage_rate":      (0.1,  0.08),
        "num_support_tickets":      (0.5,  0.8),
        "loyalty_points":           (5500, 1800),
        "weight": 0.15,
    },
}

REGIONS           = ["Western Europe", "Eastern Europe", "North America",
                     "Asia Pacific",   "Middle East",    "Latin America"]
DEVICES           = ["mobile", "desktop", "tablet"]
PAYMENT_METHODS   = ["credit_card", "paypal", "bank_transfer", "crypto", "gift_card"]
ACQ_CHANNELS      = ["organic_search", "paid_ads", "social_media",
                     "referral",        "email_campaign", "direct"]
MEMBERSHIP_TIERS  = ["none", "silver", "gold", "platinum"]

# Membership tier is correlated with segment
TIER_PROBS = {
    "Churned":  [0.60, 0.28, 0.10, 0.02],
    "At-Risk":  [0.35, 0.38, 0.22, 0.05],
    "Loyal":    [0.10, 0.30, 0.42, 0.18],
    "VIP":      [0.02, 0.08, 0.30, 0.60],
}


def _sample_numerical(profile, n, rng):
    """Draw numerical features for n customers from a segment profile."""
    rows = {}
    for feat, val in profile.items():
        if feat == "weight":
            continue
        mu, sigma = val
        if feat == "discount_usage_rate":
            vals = rng.normal(mu, sigma, n).clip(0.0, 1.0)
        elif feat in ("num_orders", "num_support_tickets"):
            vals = rng.normal(mu, sigma, n).clip(0).round().astype(int)
        elif feat in ("loyalty_points",):
            vals = rng.normal(mu, sigma, n).clip(0).round().astype(int)
        elif feat == "age":
            vals = rng.normal(mu, sigma, n).clip(18, 80).round().astype(int)
        else:
            vals = rng.normal(mu, sigma, n).clip(0)
        rows[feat] = vals
    return rows


def _sample_categorical(segment, n, rng):
    """Draw categorical features for n customers."""
    return {
        "region":             rng.choice(REGIONS,          n),
        "device_type":        rng.choice(DEVICES,          n),
        "payment_method":     rng.choice(PAYMENT_METHODS,  n),
        "acquisition_channel":rng.choice(ACQ_CHANNELS,     n),
        "membership_tier":    rng.choice(MEMBERSHIP_TIERS, n,
                                         p=TIER_PROBS[segment]),
    }


def _add_outliers(df, rng, frac=0.025):
    """
    Inject realistic outliers into ~2.5% of rows per numerical column.
    VIP-style outliers: extremely high spend / orders / loyalty points.
    """
    num_cols = ["total_spent", "num_orders", "avg_order_value", "loyalty_points",
                "avg_session_duration_min"]
    n_out = max(1, int(len(df) * frac))
    for col in num_cols:
        df[col] = df[col].astype(float)  # ensure float before assignment
        idx = rng.choice(df.index, n_out, replace=False)
        df.loc[idx, col] = df[col].mean() + rng.uniform(6, 15, n_out) * df[col].std()
    return df


def _corrupt_data(df, rng, frac=0.07):
    """
    Corrupt ~7% of cells across the dataframe:
      - 60% of corruptions -> NaN (missing values)
      - 20% -> impossible numeric values (negative age, negative spend, etc.)
      - 20% -> string sentinels in numeric columns ("N/A", "error", "-")
    """
    all_cols = df.columns.tolist()
    num_cols = ["age", "total_spent", "num_orders", "avg_order_value",
                "days_since_last_order", "avg_session_duration_min",
                "avg_pages_per_session", "discount_usage_rate",
                "num_support_tickets", "loyalty_points"]

    n_cells  = int(df.size * frac)
    rows_idx = rng.integers(0, len(df),    n_cells)
    cols_idx = rng.integers(0, len(all_cols), n_cells)

    corruption_type = rng.choice(["null", "impossible", "string"],
                                 n_cells, p=[0.60, 0.20, 0.20])

    # Cast columns that will receive string sentinels to object dtype first
    string_cols_needed = set()
    for i, ct in enumerate(corruption_type):
        col = all_cols[cols_idx[i]]
        if ct == "string" and col in num_cols:
            string_cols_needed.add(col)
    for col in string_cols_needed:
        df[col] = df[col].astype(object)

    IMPOSSIBLE = {
        "age":                      lambda: rng.choice([-5, -1, 0, 999]),
        "total_spent":              lambda: rng.uniform(-500, -1),
        "num_orders":               lambda: rng.choice([-3, -1]),
        "avg_order_value":          lambda: rng.uniform(-200, -1),
        "days_since_last_order":    lambda: rng.uniform(-100, -1),
        "avg_session_duration_min": lambda: rng.uniform(-10, -1),
        "avg_pages_per_session":    lambda: rng.uniform(-5, -1),
        "discount_usage_rate":      lambda: rng.uniform(2.0, 9.9),
        "num_support_tickets":      lambda: rng.choice([-2, -1]),
        "loyalty_points":           lambda: rng.uniform(-1000, -1),
    }
    STRING_SENTINELS = ["N/A", "error", "-", "null", "?", "##"]

    for i in range(n_cells):
        r = rows_idx[i]
        col = all_cols[cols_idx[i]]
        ct = corruption_type[i]

        if ct == "null":
            df.at[r, col] = np.nan

        elif ct == "impossible" and col in IMPOSSIBLE:
            df.at[r, col] = IMPOSSIBLE[col]()

        elif ct == "string" and col in num_cols:
            df.at[r, col] = rng.choice(STRING_SENTINELS)

    return df


def generate(n=1500, seed=42):
    """
    Generate the full e-commerce customer dataset.

    Parameters
    ----------
    n    : total number of customers
    seed : random seed for reproducibility

    Returns
    -------
    pd.DataFrame  with n rows and 16 columns (15 features + target)
    """
    rng = np.random.default_rng(seed)

    segments = list(SEGMENT_PROFILES.keys())
    weights  = [SEGMENT_PROFILES[s]["weight"] for s in segments]
    counts   = np.round(np.array(weights) * n).astype(int)
    # Fix rounding so total == n
    counts[-1] += n - counts.sum()

    frames = []
    for seg, cnt in zip(segments, counts):
        profile = SEGMENT_PROFILES[seg]
        num  = _sample_numerical(profile, cnt, rng)
        cat  = _sample_categorical(seg, cnt, rng)
        df_s = pd.DataFrame({**num, **cat})
        df_s["customer_segment"] = seg
        frames.append(df_s)

    df = pd.concat(frames, ignore_index=True)

    # Shuffle rows
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Add a synthetic customer_id
    df.insert(0, "customer_id", [f"CUST{str(i).zfill(5)}" for i in range(1, len(df)+1)])

    # Outliers before corruption (so some outliers may also get corrupted)
    df = _add_outliers(df, rng)

    # Corrupt a fraction of cells
    df = _corrupt_data(df, rng)

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate e-commerce customer dataset")
    parser.add_argument("--n",    type=int, default=1500, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42,   help="Random seed")
    parser.add_argument("--out",  type=str, default="ecommerce_customers.csv",
                        help="Output CSV file path")
    args = parser.parse_args()

    df = generate(n=args.n, seed=args.seed)
    df.to_csv(args.out, index=False)

    print(f"Dataset saved to: {args.out}")
    print(f"Shape            : {df.shape}")
    print(f"\nClass distribution:\n{df['customer_segment'].value_counts()}")
    print(f"\nMissing values per column:\n{df.isnull().sum()}")
    print(f"\nSample rows:\n{df.head(3).to_string()}")
