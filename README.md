# Benchmarking Anomaly Detection on Ethereum: Unsupervised vs. Semi-Supervised vs. Supervised Paradigms

A reproducible empirical benchmark comparing **unsupervised**, **semi-supervised**, and **supervised** machine learning paradigms for account-level fraud and anomaly detection on the Ethereum blockchain.

All experiments use identical data cleaning, identical cross-validation splits, and strictly isolated preprocessing to ensure an apples-to-apples comparison.

---

## Overview & Motivation

Ground-truth labels on public blockchains are scarce, delayed, and expensive to verify. While academic literature frequently benchmarks standard supervised models on blockchain datasets, production systems often face scenarios where:
1. **No labels exist** (Unsupervised anomaly detection).
2. **Only benign traffic is readily available** (Semi-supervised novelty detection).
3. **Only a tiny fraction of accounts have verified labels** (Semi-supervised self-training vs. inductive few-shot supervision).

This repository systematically evaluates eight distinct model configurations across these learning paradigms using the Kaggle **Ethereum Fraud Detection** benchmark (`transaction_dataset.csv`).

---

## Key Experimental Findings

1. **Supervised ceilings (~0.99 ROC-AUC)**: Decision tree ensembles achieve near-perfect discrimination. Feature importance analysis proves this is largely driven by account lifespan (`Time Diff between first and last`) and fund drainage patterns, reflecting the disposable "burner wallet" nature of curated fraud datasets.
2. **The Self-Training Pitfall**: When only 5% of training labels are available, an inductive Random Forest trained *strictly on the labeled fraction* (`RF-Only`) achieves **0.954 ROC-AUC**, outperforming self-training pseudo-labeling (**0.934 ROC-AUC**). Pseudo-labeling ambiguous boundary instances induces confirmation bias that degrades decision trees.
3. **Novelty Detection vs. Pure Unsupervised**: Semi-supervised novelty detection on legitimate accounts (Autoencoder ROC-AUC **0.802**) vastly outperforms fully unsupervised training on mixed data (ROC-AUC **0.667**). Training unsupervised reconstruction models on uncurated data containing fraud contaminates the latent space and teaches the network to reconstruct fraudulent patterns.
4. **Metric Integrity**: Due to class imbalance (17.8% fraud), relying solely on ROC-AUC paints an overly optimistic picture. We report **Average Precision (PR-AUC)**, **R-Precision**, and **Oracle Best-F1** alongside fixed-threshold metrics.

---

## Repository Structure

```text
├── ethereum_common.py         # Shared pipeline: loading, de-duplication, leakage prevention, metrics
├── ethereum_supervised.py     # Supervised models: Random Forest & HistGradientBoosting
├── ethereum_semisupervised.py # Setting A (normal-only novelty) & Setting B (self-training vs. RF-only)
├── ethereum_unsupervised.py   # Unsupervised models: Isolation Forest, PCA, Autoencoder
├── requirements.txt           # Python dependency specifications
└── transaction_dataset.csv    # Kaggle dataset (place in root directory)
```

---

## Leakage Prevention & Data Hygiene Safeguards

Prior benchmarks on this dataset often report inflated scores due to subtle data collection artifacts. This codebase enforces several explicit safeguards in `ethereum_common.py`:

* **ERC-20 Missing Value Leak**: In the raw dataset, missing ERC-20 attributes were left blank (`NaN`) for 549 fraud accounts but represented as `0` for benign accounts. We explicitly impute missing ERC-20 values to `0` across all classes, eliminating the trivial artifact leak.
* **Scam Campaign Memorization**: Specific token name columns (`ERC20 most sent token type`, `ERC20_most_rec_token_type`) contained high-cardinality strings from a single 2018 phishing campaign. These text columns are automatically dropped (`DROP_TOKEN_NAMES = True`) while preserving valid numerical token count features (`ERC20 uniq sent/rec token name`).
* **Contradictory & Duplicate Account Handling**: 
  * 25 duplicate addresses are removed.
  * 528 rows with identical feature vectors are de-duplicated prior to splitting.
  * Rows with identical features but contradictory ground-truth labels are detected and filtered out as ambiguous noise.
* **Zero-Variance Feature Pruning**: 7 constant columns (e.g., contract transfer timings with zero variance) are dropped to prevent numerical instability in linear algebra and neural routines.
* **Leakage-Free Preprocessing**: Imputation (`SimpleImputer`) and non-linear scaling (`QuantileTransformer` to normal distribution) are fitted strictly on training accounts and applied transductively to test accounts. For novelty detection (Setting A), preprocessing is fitted strictly on legitimate training accounts.

---

## Paradigm Summary & Evaluation Matrix

| Paradigm | Model | Training Input | Decision Rule |
| :--- | :--- | :--- | :--- |
| **Supervised** | Random Forest / HistGradientBoosting | 100% of training labels | Probability $\ge 0.5$ |
| **Semi-Supervised (B)** | `RF-Only` (5%, 10%, 20% labels) | Only visible labeled subset | Probability $\ge 0.5$ |
| **Semi-Supervised (B)** | `SelfTraining-RF` (5%, 10%, 20% labels) | Visible labels + confident pseudo-labels | Probability $\ge 0.5$ |
| **Semi-Supervised (A)** | Autoencoder / PCA / Isolation Forest | Normal (legitimate) training accounts only | Top 5% anomaly scores on normal training data |
| **Unsupervised** | Autoencoder / PCA / Isolation Forest | Unlabeled mixture of normal & fraud accounts | Top 5% anomaly scores on mixed training data |

---

## Benchmark Results

All models were evaluated across **5 stratified random splits** (`SEEDS = [42, 43, 44, 45, 46]`) with a 70/30 train/test partition (6,501 train / 2,787 test). Results reflect **Mean ± Standard Deviation**:

| Paradigm | Model / Configuration | ROC-AUC | PR-AUC (AP) | R-Precision | Best-F1 | Precision@1% |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Supervised** | HistGradientBoosting (100% labels) | **0.990 ± 0.001** | **0.968 ± 0.004** | **0.903 ± 0.007** | **0.912 ± 0.008** | **1.000 ± 0.000** |
| **Supervised** | Random Forest (100% labels) | **0.987 ± 0.002** | **0.958 ± 0.008** | **0.888 ± 0.009** | **0.894 ± 0.007** | **1.000 ± 0.000** |
| **Semi-Supervised (B)** | `RF-Only` (20% labels) | **0.978 ± 0.004** | **0.927 ± 0.012** | **0.841 ± 0.020** | **0.847 ± 0.020** | **1.000 ± 0.000** |
| **Semi-Supervised (B)** | `SelfTraining-RF` (20% labels) | **0.974 ± 0.005** | **0.915 ± 0.011** | **0.823 ± 0.019** | **0.827 ± 0.018** | **0.985 ± 0.020** |
| **Semi-Supervised (B)** | `RF-Only` (10% labels) | **0.967 ± 0.005** | **0.895 ± 0.018** | **0.799 ± 0.016** | **0.807 ± 0.015** | **1.000 ± 0.000** |
| **Semi-Supervised (B)** | `SelfTraining-RF` (10% labels) | **0.959 ± 0.007** | **0.867 ± 0.026** | **0.775 ± 0.024** | **0.779 ± 0.022** | **0.993 ± 0.017** |
| **Semi-Supervised (B)** | `RF-Only` (5% labels) | **0.954 ± 0.013** | **0.873 ± 0.029** | **0.782 ± 0.028** | **0.791 ± 0.026** | **1.000 ± 0.000** |
| **Semi-Supervised (B)** | `SelfTraining-RF` (5% labels) | **0.934 ± 0.015** | **0.827 ± 0.038** | **0.748 ± 0.027** | **0.755 ± 0.025** | **0.985 ± 0.033** |
| **Semi-Supervised (A)** | Autoencoder (normal-only) | **0.802 ± 0.023** | **0.384 ± 0.043** | **0.420 ± 0.068** | **0.510 ± 0.029** | **0.252 ± 0.096** |
| **Semi-Supervised (A)** | PCA (normal-only) | **0.753 ± 0.005** | **0.333 ± 0.013** | **0.343 ± 0.014** | **0.453 ± 0.005** | **0.259 ± 0.069** |
| **Semi-Supervised (A)** | Isolation Forest (normal-only) | **0.738 ± 0.011** | **0.345 ± 0.019** | **0.341 ± 0.028** | **0.448 ± 0.012** | **0.400 ± 0.103** |
| **Unsupervised** | Autoencoder (mixed training) | **0.667 ± 0.014** | **0.256 ± 0.015** | **0.285 ± 0.024** | **0.383 ± 0.011** | **0.148 ± 0.059** |
| **Unsupervised** | PCA (mixed training) | **0.650 ± 0.010** | **0.260 ± 0.014** | **0.312 ± 0.022** | **0.364 ± 0.004** | **0.185 ± 0.026** |
| **Unsupervised** | Isolation Forest (mixed training) | **0.637 ± 0.020** | **0.261 ± 0.018** | **0.252 ± 0.024** | **0.369 ± 0.012** | **0.311 ± 0.093** |

*Note: Baseline random guessing yield for AP equals the fraud prevalence of 0.1783.*

---

## Installation & Reproducibility

### 1. Requirements

Ensure you have Python 3.9+ installed. Install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Dataset Setup

Download `transaction_dataset.csv` from Kaggle's [Ethereum Fraud Detection Dataset](https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset) and place the file in the project root directory.

### 3. Running the Experiments

To run each experiment across all 5 random splits:

```bash
# Supervised experiments (Random Forest & Gradient Boosting)
python ethereum_supervised.py

# Unsupervised experiments (Isolation Forest, PCA, Autoencoder)
python ethereum_unsupervised.py

# Semi-supervised experiments (Novelty detection & Self-training vs. RF-only)
python ethereum_semisupervised.py
```

Each script automatically exports:
* `*_results_by_seed.csv`: Granular per-seed metric evaluation.
* `*_results_summary.csv`: Aggregated mean and standard deviation.
* `*_scores.csv`: Account-level probability/anomaly predictions on the test set for plotting ROC and PR curves.
* `*_autoencoder_log.csv`: Training loss histories and convergence logs for deep models.
* `*_rf_feature_importances.csv`: Feature importance rankings (Supervised).

---

## Critical Methodological Considerations for Researchers

When citing or building upon these benchmarks in academic research, consider the following structural constraints of account-level blockchain tabular data:

1. **Curated Blacklist Sampling Bias**: Fraudulent wallets in public datasets are sourced from external scam databases (e.g., Etherscan phishing lists). These addresses typically represent single-purpose fraud campaigns that disperse funds rapidly, creating distinct separation from long-standing benign accounts.
2. **Aggregated Lifetime Features (Look-Ahead Bias)**: Attributes such as `Time Diff between first and last` and `total ether balance` summarize an account's complete transaction history in hindsight. Live production systems must classify transactions incrementally.
3. **Threshold vs. Ranking Metrics**: In imbalanced fraud detection, fixed-threshold F1 scores are heavily biased by the chosen decision threshold. We strongly advise evaluating models on **Average Precision (AP / PR-AUC)** and **R-Precision**, which evaluate the model's intrinsic ranking quality regardless of operational alert thresholds.