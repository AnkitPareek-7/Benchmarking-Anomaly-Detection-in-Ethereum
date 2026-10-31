"""
Semi-supervised anomaly detection on the Ethereum fraud dataset.

Two semi-supervised settings:

  A) Normal-only training (novelty detection)
     Models learn what "legitimate" looks like from legitimate training
     accounts only. Anything that deviates from it is anomalous. The
     preprocessing is also fit on legitimate accounts only, so no fraud
     account influences the model in any way.
     Models: Isolation Forest, PCA, Autoencoder (same as the unsupervised
     experiment, for a fair comparison).

  B) Self-training with few labels vs. Supervised-on-subset baselines
     Only a small fraction of training labels (5%, 10%, 20%) stays visible;
     the rest is treated as unlabeled. We evaluate:
       1. RF-Only: supervised Random Forest trained strictly on visible labels
       2. SelfTraining-RF: pseudo-labels confident unlabeled accounts and retrains
     Comparing (2) against (1) measures the true lift provided by self-training.

Same data and splits as the other two experiments.

Usage:  python ethereum_semisupervised.py [--split random|grouped]
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.semi_supervised import SelfTrainingClassifier

from ethereum_common import (
    SEEDS, setup, split_data, preprocess, evaluate, summarize,
)

# Setting A: flag accounts above this quantile of normal-training scores
NORMAL_FLAG_QUANTILE = 0.95

# Setting B: fractions of training labels that stay visible
LABELED_FRACTIONS = [0.05, 0.10, 0.20]

X, y, groups, SPLIT = setup("Semi-supervised anomaly detection")
PREFIX = f"ethereum_semisupervised_{SPLIT}"

# Autoencoder training log: epochs run before early stopping, per split
AE_LOG = []


# ======================================================================
# SETTING A: normal-only training
# ======================================================================
def isolation_forest(X_normal, X_test, seed):
    model = IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1)
    model.fit(X_normal)
    return -model.decision_function(X_normal), -model.decision_function(X_test)


def pca_reconstruction(X_normal, X_test, seed):
    pca = PCA(n_components=0.95, random_state=seed).fit(X_normal)

    def error(A):
        A_hat = pca.inverse_transform(pca.transform(A))
        return np.mean((A - A_hat) ** 2, axis=1)

    return error(X_normal), error(X_test)


def autoencoder(X_normal, X_test, seed):
    import tensorflow as tf
    from tensorflow import keras

    # Clear previous Keras session graph to prevent memory bloat across seeds
    keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    d = X_normal.shape[1]

    inp = keras.Input(shape=(d,))
    x = keras.layers.Dense(64, activation="relu")(inp)
    z = keras.layers.Dense(16, activation="relu")(x)
    x = keras.layers.Dense(64, activation="relu")(z)
    out = keras.layers.Dense(d, activation="linear")(x)

    ae = keras.Model(inp, out)
    ae.compile(optimizer="adam", loss="mse")

    # Hold out 10% of the normal rows for early stopping
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_normal))
    n_val = max(1, int(0.1 * len(X_normal)))
    X_val, X_fit = X_normal[idx[:n_val]], X_normal[idx[n_val:]]

    history = ae.fit(
        X_fit, X_fit,
        validation_data=(X_val, X_val),
        epochs=50, batch_size=512, shuffle=True, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True)],
    )

    val_loss = history.history["val_loss"]
    AE_LOG.append({"seed": seed,
                   "epochs_run": len(val_loss),
                   "best_epoch": int(np.argmin(val_loss)) + 1,
                   "best_val_loss": float(np.min(val_loss)),
                   "stopped_early": len(val_loss) < 50})
    print(f"   Autoencoder: {len(val_loss)} epochs run, "
          f"best epoch {int(np.argmin(val_loss)) + 1}")

    def error(A):
        A_hat = ae.predict(A, batch_size=2048, verbose=0)
        return np.mean((A - A_hat) ** 2, axis=1)

    return error(X_normal), error(X_test)


NORMAL_ONLY_MODELS = {
    "IsolationForest (normal-only)": isolation_forest,
    "PCA (normal-only)": pca_reconstruction,
    "Autoencoder (normal-only)": autoencoder,
}


# ======================================================================
# SETTING B: self-training with a small labeled subset
# ======================================================================
def hide_labels(y_true, labeled_fraction, seed):
    """Keep `labeled_fraction` of the labels in each class; hide the rest (-1)."""
    rng = np.random.default_rng(seed)
    y_partial = np.full(len(y_true), -1)
    for cls in (0, 1):
        idx = np.where(y_true == cls)[0]
        n_keep = max(1, int(labeled_fraction * len(idx)))
        y_partial[rng.choice(idx, n_keep, replace=False)] = cls
    return y_partial


# ----------------------------------------------------------------------
# Run everything on every split
# ----------------------------------------------------------------------
rows = []
scores_first_seed, first_split = {}, None

for seed in SEEDS:
    X_train, X_test, y_train, y_test = split_data(X, y, groups, seed, SPLIT)
    print(f"\n--- Seed {seed}: train {len(X_train)} | test {len(X_test)} | "
          f"test fraud {y_test.mean():.4f} ---")
    if seed == SEEDS[0]:
        first_split = (X_test.index, y_test)

    # Setting A: only legitimate training accounts are used, including for
    # fitting the preprocessing
    X_normal_raw = X_train[y_train == 0]
    X_normal, Xte_a, _ = preprocess(X_normal_raw, X_test, seed)
    for name, run in NORMAL_ONLY_MODELS.items():
        print(f"Running {name}...")
        scores_normal, scores_test = run(X_normal, Xte_a, seed)
        threshold = np.quantile(scores_normal, NORMAL_FLAG_QUANTILE)

        rows.append({"dataset": "Ethereum_Fraud_Detection", "paradigm": "semi-supervised",
                     "setting": "A: normal-only", "split": SPLIT, "seed": seed,
                     "model": name, **evaluate(y_test, scores_test, threshold)})
        if seed == SEEDS[0]:
            scores_first_seed[name] = scores_test

    # Setting B: a few labels + many unlabeled training accounts
    Xtr, Xte, _ = preprocess(X_train, X_test, seed)
    for frac in LABELED_FRACTIONS:
        pct = int(frac * 100)
        y_partial = hide_labels(y_train, frac, seed)
        labeled_mask = (y_partial != -1)

        # 1. Inductive Baseline: Random Forest trained strictly on the labeled fraction
        print(f"Running RF-Only ({pct}% labels)...")
        rf_base = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                         random_state=seed, n_jobs=-1)
        rf_base.fit(Xtr[labeled_mask], y_partial[labeled_mask])
        base_scores = rf_base.predict_proba(Xte)[:, 1]

        rows.append({"dataset": "Ethereum_Fraud_Detection", "paradigm": "semi-supervised",
                     "setting": "B: baseline (labeled only)", "split": SPLIT, "seed": seed,
                     "model": f"RF-Only ({pct}% labels)",
                     **evaluate(y_test, base_scores, threshold=0.5)})

        # 2. Semi-Supervised Self-Training RF
        print(f"Running SelfTraining-RF ({pct}% labels)...")
        st_base = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                         random_state=seed, n_jobs=-1)
        model = SelfTrainingClassifier(st_base, threshold=0.90, max_iter=10)
        model.fit(Xtr, y_partial)

        scores_test = model.predict_proba(Xte)[:, 1]      # probability of fraud

        rows.append({"dataset": "Ethereum_Fraud_Detection", "paradigm": "semi-supervised",
                     "setting": "B: self-training", "split": SPLIT, "seed": seed,
                     "model": f"SelfTraining-RF ({pct}% labels)",
                     **evaluate(y_test, scores_test, threshold=0.5)})

        if seed == SEEDS[0]:
            scores_first_seed[f"RF-Only ({pct}% labels)"] = base_scores
            scores_first_seed[f"SelfTraining-RF ({pct}% labels)"] = scores_test

results = pd.DataFrame(rows)
summarize(results, PREFIX)

# Account-level scores of the first split (handy for ROC / PR plots)
score_table = pd.DataFrame({"original_row_index": first_split[0],
                            "actual_label": first_split[1]})
for name, scores in scores_first_seed.items():
    score_table[f"{name}_score"] = scores
score_table.to_csv(f"{PREFIX}_scores.csv", index=False)
pd.DataFrame(AE_LOG).to_csv(f"{PREFIX}_autoencoder_log.csv", index=False)
print(f"- {PREFIX}_scores.csv (first split only)")
print(f"- {PREFIX}_autoencoder_log.csv (epochs per split)")
