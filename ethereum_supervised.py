"""
Supervised fraud detection on the Ethereum fraud dataset.

Two supervised models are trained with ALL training labels:
  1. Random Forest       (bagging of decision trees)
  2. Gradient Boosting   (sequential boosting of decision trees)

Same data, splits and preprocessing as the unsupervised and semi-supervised
experiments, repeated over several splits.

Usage:  python ethereum_supervised.py [--split random|grouped]
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from ethereum_common import SEEDS, setup, split_and_preprocess, evaluate, summarize

DECISION_THRESHOLD = 0.5      # fraud probability above which an account is flagged

X, y, groups, SPLIT = setup("Supervised fraud detection")
PREFIX = f"ethereum_supervised_{SPLIT}"


def make_models(seed):
    # class_weight="balanced" makes the rarer fraud class count more in training
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        ),
        "GradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, class_weight="balanced",
            random_state=seed,
        ),
    }


rows = []
scores_first_seed, first_split = {}, None

for seed in SEEDS:
    Xtr, Xte, y_train, y_test, test_index, feature_names = split_and_preprocess(
        X, y, groups, seed, SPLIT
    )
    print(f"\n--- Seed {seed}: train {len(Xtr)} | test {len(Xte)} | "
          f"test fraud {y_test.mean():.4f} ---")

    for name, model in make_models(seed).items():
        print(f"Training {name}...")
        model.fit(Xtr, y_train)
        scores_test = model.predict_proba(Xte)[:, 1]      # probability of fraud

        rows.append({"dataset": "Ethereum_Fraud_Detection", "paradigm": "supervised",
                     "split": SPLIT, "seed": seed, "model": name,
                     **evaluate(y_test, scores_test, DECISION_THRESHOLD)})

        if seed == SEEDS[0]:
            scores_first_seed[name] = scores_test
            first_split = (test_index, y_test)
            if name == "RandomForest":
                fi_df = pd.DataFrame({
                    "feature": feature_names,
                    "importance": model.feature_importances_,
                }).sort_values("importance", ascending=False)
                fi_df.to_csv(f"{PREFIX}_rf_feature_importances.csv", index=False)
                print(f"   Saved {PREFIX}_rf_feature_importances.csv")
                print("   Top 5 features:\n   ",
                      fi_df.head(5).to_string(index=False).replace("\n", "\n   "))

results = pd.DataFrame(rows)
summarize(results, PREFIX)

# Account-level scores of the first split (handy for ROC / PR plots)
score_table = pd.DataFrame({"original_row_index": first_split[0],
                            "actual_label": first_split[1]})
for name, scores in scores_first_seed.items():
    score_table[f"{name}_score"] = scores
score_table.to_csv(f"{PREFIX}_scores.csv", index=False)
print(f"- {PREFIX}_scores.csv (first split only)")
