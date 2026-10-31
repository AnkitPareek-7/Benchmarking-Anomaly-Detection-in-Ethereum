"""
Shared helpers for the three Ethereum fraud-detection experiments
(unsupervised, semi-supervised and supervised).

Data loading, cleaning, splitting, preprocessing and evaluation live here so
that all three paradigms see exactly the same data and the same splits.

Safeguards & Methodological Standards:
  1. Identifiers and duplicate addresses are removed.
  2. Missing-value encodings are harmonised: In the raw file, ERC20 columns
     are blank ONLY for fraud accounts (549 fraud, 0 legitimate), while
     legitimate accounts without ERC20 activity are stored as 0. We encode
     "no ERC20 data" as 0 for all classes to eliminate this artifact leak.
  3. High-cardinality token-name columns are dropped by default to prevent
     models from memorizing a single scam campaign name.
  4. Contradictory rows (identical features with conflicting labels) are
     detected and removed as ambiguous ground-truth noise. Duplicate rows
     with matching labels are de-duplicated prior to splitting.
  5. Preprocessing is strictly fit on training data only.
  6. Standard evaluation metrics include ROC-AUC, Average Precision (AP),
     R-Precision, Precision@1%, and Oracle Best-F1.
"""

import argparse

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)

# ----------------------------------------------------------------------
# Settings shared by all experiments
# ----------------------------------------------------------------------
DATA_PATH = "transaction_dataset.csv"
TEST_SIZE = 0.30                      # "random" protocol
SEEDS = [42, 43, 44, 45, 46]          # one split per seed

DROP_TOKEN_NAMES = True               # drop scam token name columns
N_GROUPS = 100                        # behaviour clusters for the "grouped" protocol
GROUP_FOLDS = 3                       # "grouped": one fold (~1/3) is the test set

# Metrics reported in the mean +/- std summary
METRICS = [
    "ROC_AUC",
    "AP",
    "R_Precision",
    "Precision@1%",
    "Best_F1",
    "Precision",
    "Recall",
    "F1",
]


# ----------------------------------------------------------------------
# 1. Load, clean and de-duplicate
# ----------------------------------------------------------------------
def _harmonise_missing(X):
    """
    Encode 'no ERC20 data' identically for every class, coerce numeric columns,
    and drop high-cardinality token names.
    """
    X = X.copy()

    # Text columns holding the NAME of the most-used token ("... token type").
    # Both words are required: the numeric counts "ERC20 uniq sent/rec token
    # name" are ordinary behavioural features and must be kept.
    token_cols = [c for c in X.columns
                  if "token" in c.lower() and "type" in c.lower()]

    # Coerce all non-token columns to numeric (handles strings, whitespaces, 'None')
    for c in X.columns:
        if c not in token_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # In ERC20 columns, fill NaNs with 0 (removes the blank-for-fraud-only leak)
    erc20_cols = [c for c in X.columns if "erc20" in c.lower() and c not in token_cols]
    X[erc20_cols] = X[erc20_cols].fillna(0)

    # Handle token columns
    if DROP_TOKEN_NAMES and token_cols:
        X = X.drop(columns=token_cols)
        dropped_tokens = token_cols
    else:
        for c in token_cols:
            s = X[c].astype(str).str.strip().str.lower()
            X[c] = s.where(~s.isin(["0", "", "nan", "none"]), "none")
        dropped_tokens = []

    return X, len(erc20_cols), dropped_tokens


def load_data(path=DATA_PATH):
    """Return the feature table X and binary label y (1 = fraud)."""
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()

    target_col = next((c for c in df.columns if c.lower() == "flag"), None)
    if target_col is None:
        raise ValueError("Could not find the FLAG column.")

    df = df.dropna(axis=1, how="all")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col]).copy()
    df[target_col] = df[target_col].astype(int)

    # Identifier columns are not behavioural features
    id_names = {"index", "address", "wallet", "wallet_address",
                "txid", "tx_id", "transaction_hash", "hash"}
    drop_cols = [
        c for c in df.columns
        if c != target_col
        and (c.lower() in id_names or c.lower().startswith("unnamed:"))
    ]
    address_col = next((c for c in df.columns if c.lower() == "address"), None)

    n_raw = len(df)

    # (a) Drop exact repeat wallet addresses
    if address_col is not None:
        df = df.drop_duplicates(subset=address_col, keep="first")
    n_after_address = len(df)

    # (b) Harmonise missing-value encodings & drop token names
    features_raw = df.drop(columns=[target_col] + drop_cols)
    X_cleaned, n_erc20, dropped_tokens = _harmonise_missing(features_raw)
    y_raw = df[target_col]

    # (c) Drop zero-variance (constant across all accounts) features
    const_cols = [c for c in X_cleaned.columns if X_cleaned[c].nunique(dropna=False) <= 1]
    if const_cols:
        X_cleaned = X_cleaned.drop(columns=const_cols)

    # (d) Detect and handle identical feature rows safely
    feature_cols = list(X_cleaned.columns)
    full_df = X_cleaned.copy()
    full_df["_TARGET_LABEL_"] = y_raw

    # Flag contradictory rows: identical features but differing labels
    contradictory_mask = (
        full_df.duplicated(subset=feature_cols, keep=False) &
        ~full_df.duplicated(subset=feature_cols + ["_TARGET_LABEL_"], keep=False)
    )
    n_contradictory = int(contradictory_mask.sum())
    if n_contradictory > 0:
        full_df = full_df[~contradictory_mask]

    # Drop duplicate identical rows (identical features AND identical label)
    n_before_dedup = len(full_df)
    full_df = full_df.drop_duplicates(subset=feature_cols, keep="first")
    n_dedup = n_before_dedup - len(full_df)

    X = full_df[feature_cols].copy()
    y = full_df["_TARGET_LABEL_"].copy()

    print(f"Rows in file: {n_raw}")
    print(f"Removed duplicate addresses: {n_raw - n_after_address}")
    if n_contradictory > 0:
        print(f"Removed contradictory rows (same features, opposite label): {n_contradictory}")
    print(f"Removed identical feature rows: {n_dedup}")
    print(f"Rows kept: {len(X)} | Feature count: {X.shape[1]}")
    print(f"Legitimate accounts: {int((y == 0).sum())}")
    print(f"Fraudulent accounts: {int((y == 1).sum())}")
    print(f"Fraud prevalence: {round(y.mean(), 4)}")
    print(f"Removed identifier columns: {drop_cols}")
    if const_cols:
        print(f"Removed constant zero-variance columns: {const_cols}")
    print(f"Missing ERC20 values set to 0 in {n_erc20} columns; "
          f"token-name columns {'dropped' if DROP_TOKEN_NAMES else 'harmonised'}: {dropped_tokens}")
    return X, y


# ----------------------------------------------------------------------
# 2. Groups of similar accounts (for the "grouped" protocol)
# ----------------------------------------------------------------------
def make_groups(X, seed=0):
    """
    Cluster accounts by behaviour. Whole clusters are held out together
    in the 'grouped' protocol as a synthetic stress-test.
    """
    numeric = X.select_dtypes(include=np.number)
    Z = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("quantile", QuantileTransformer(
            n_quantiles=min(1000, len(X)), output_distribution="normal",
            random_state=seed)),
    ]).fit_transform(numeric)
    return KMeans(N_GROUPS, n_init=3, random_state=seed).fit_predict(Z)


# ----------------------------------------------------------------------
# 3. Splitting and preprocessing
# ----------------------------------------------------------------------
def split_data(X, y, groups, seed, mode):
    """Returns raw X_train, X_test and numpy y_train, y_test."""
    if mode == "random":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed,
            shuffle=True, stratify=y,
        )
    elif mode == "grouped":
        cv = StratifiedGroupKFold(n_splits=GROUP_FOLDS, shuffle=True,
                                  random_state=seed)
        train_idx, test_idx = next(cv.split(X, y, groups))
        assert set(groups[train_idx]).isdisjoint(groups[test_idx])
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        raise ValueError("mode must be 'random' or 'grouped'")
    return X_train, X_test, y_train.to_numpy(), y_test.to_numpy()


def preprocess(X_fit, X_apply, seed):
    """
    Fit preprocessing on X_fit only and apply it to both tables.
    Returns:
      A (processed X_fit), B (processed X_apply), feature_names
    """
    numeric_cols = X_fit.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in X_fit.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("quantile", QuantileTransformer(
            n_quantiles=min(1000, len(X_fit)),
            output_distribution="normal",
            random_state=seed,
        )),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipe, numeric_cols))
    if categorical_cols:
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ])
        transformers.append(("categorical", categorical_pipe, categorical_cols))

    preprocessor = ColumnTransformer(transformers, remainder="drop",
                                     sparse_threshold=0.0)

    A = np.asarray(preprocessor.fit_transform(X_fit), dtype=np.float32)
    B = np.asarray(preprocessor.transform(X_apply), dtype=np.float32)

    if not np.isfinite(A).all() or not np.isfinite(B).all():
        raise ValueError("Non-finite values remain after preprocessing.")

    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"f_{i}" for i in range(A.shape[1])]

    return A, B, feature_names


def split_and_preprocess(X, y, groups, seed, mode):
    """
    Split, then preprocess (fit on the training part only).
    Returns Xtr, Xte, y_train, y_test, test_index, feature_names.
    """
    X_train, X_test, y_train, y_test = split_data(X, y, groups, seed, mode)
    Xtr, Xte, feature_names = preprocess(X_train, X_test, seed)
    return Xtr, Xte, y_train, y_test, X_test.index, feature_names


# ----------------------------------------------------------------------
# 4. Evaluation (higher score = more likely fraud)
# ----------------------------------------------------------------------
def evaluate(y_test, scores, threshold):
    """
    Evaluate ranking metrics on raw continuous scores and classification metrics
    at `scores >= threshold`. Also computes Oracle Best-F1 and operational review budgets.
    """
    prevalence = float(y_test.mean())
    pred = (scores >= threshold).astype(int)

    # 1. R-Precision (Precision@k where k = total true positive instances in test)
    k = max(1, int(y_test.sum()))
    top_k_indices = np.argsort(-scores)[:k]
    r_precision = float(y_test[top_k_indices].mean())

    # 2. Precision@1% (top 1% operational inspection budget)
    k_1pct = max(1, int(0.01 * len(y_test)))
    top_1pct_indices = np.argsort(-scores)[:k_1pct]
    precision_1pct = float(y_test[top_1pct_indices].mean())

    # 3. Average Precision (PR-AUC)
    ap = float(average_precision_score(y_test, scores))

    # 4. Oracle Best-F1 (upper-bound F1 achievable across all possible thresholds)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, scores)
    denom = prec_curve + rec_curve
    f1_curve = np.divide(2 * prec_curve * rec_curve, denom,
                         out=np.zeros_like(denom), where=denom > 0)
    best_f1 = float(np.max(f1_curve))

    return {
        "ROC_AUC": float(roc_auc_score(y_test, scores)),
        "AP": ap,
        "AP_lift_vs_random": float(ap / max(1e-6, prevalence)),
        "R_Precision": r_precision,
        "Precision@k": r_precision,  # backward compatibility
        "Precision@1%": precision_1pct,
        "Best_F1": best_f1,
        "Threshold": float(threshold),
        "Flagged_fraction": float(pred.mean()),
        "Precision": float(precision_score(y_test, pred, zero_division=0)),
        "Recall": float(recall_score(y_test, pred, zero_division=0)),
        "F1": float(f1_score(y_test, pred, zero_division=0)),
    }


# ----------------------------------------------------------------------
# 5. Command line, setup and summary
# ----------------------------------------------------------------------
def setup(description):
    """Parse --split, load the data and (for 'grouped') build the groups."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--split", choices=["random", "grouped"], default="random",
                        help="random: stratified random split (default, recommended); "
                             "grouped: similar accounts are held out together (stress-test)")
    split = parser.parse_args().split

    X, y = load_data()
    groups = make_groups(X) if split == "grouped" else None
    print(f"\nSplit protocol: {split}")
    return X, y, groups, split


def summarize(results, prefix):
    """
    Save per-seed results and a mean/std summary, and print the summary.
    `results` has one row per (seed, model).
    """
    results.to_csv(f"{prefix}_results_by_seed.csv", index=False)

    grouped = results.groupby("model", sort=False)[METRICS]
    mean, std = grouped.mean(), grouped.std()

    mean.join(std, lsuffix="_mean", rsuffix="_std").to_csv(
        f"{prefix}_results_summary.csv"
    )

    pretty = mean.round(4).astype(str) + " +/- " + std.round(4).astype(str)
    n_seeds = results["seed"].nunique()
    print(f"\n=== Mean +/- std over {n_seeds} splits ===")
    print(pretty.to_string())

    print("\nSaved:")
    print(f"- {prefix}_results_by_seed.csv")
    print(f"- {prefix}_results_summary.csv")
