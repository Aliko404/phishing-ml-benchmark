# Machine Learning Approaches for Phishing Website Detection: A Comparative Performance Analysis

**Authors:** Alinur Bedelkhan, Agadilkhan Nurdaulet
**Affiliation:** Narxoz University
**Status:** Week 6 — measurement pipeline ready, smoke-tested on a micro sample.

## Overview

Phishing websites are created and modified faster than blacklists can follow, which makes
machine learning an attractive detection approach. Existing studies often report high accuracy
but use different datasets, feature sets and splits, and rarely report computational cost
or the effect of feature selection together. This repository contains a reproducible benchmark
that compares classical ML classifiers on **one dataset, one feature set and one split**, and
measures how **feature selection** changes detection quality and inference cost.

## Research Question

> **RQ:** How do classical machine learning algorithms compare in phishing website detection
> (accuracy, precision, recall, F1, inference time), and how does feature selection affect
> their performance and computational efficiency?

Sub-questions: RQ1 (achievable accuracy), RQ2 (algorithm comparison), RQ3 (effect of feature selection).
The full methodology (hypotheses, variable matrix, metrics, baseline) is in
[`docs/METHODOLOGY_CARD.md`](docs/METHODOLOGY_CARD.md).

## Planned Benchmarks

| ID | Independent variable | Levels | Primary output |
|----|----------------------|--------|----------------|
| B1 | ML algorithm | Logistic Regression (baseline), Decision Tree, Random Forest, SVM, KNN, Gradient Boosting, AdaBoost | F1, accuracy, precision, recall |
| B2 | ML algorithm | same as B1 | inference time (ms/sample), train time, peak RAM |
| B3 | Feature selection | none, mutual information (top-k), RF importance (top-k), PCA | ΔF1 and Δinference time vs. "none" |

Each (algorithm × feature selection) pair is one run: 7 × 4 = 28 runs per dataset.

## System Requirements

- Docker 24+ **or** Python 3.13
- ≥ 2 GB RAM, ≥ 1 CPU core (the sample run takes ~10 s)
- Linux / macOS / Windows (WSL2 for Docker)

## Quickstart

With Docker (3 commands):

```bash
git clone https://github.com/Aliko404/phishing-ml-benchmark.git && cd phishing-ml-benchmark
docker build -t phishing-bench .
docker run --rm -v "$(pwd)/results:/app/results" phishing-bench
```

Without Docker:

```bash
pip install -r requirements.txt   # or: conda env create -f environment.yml
python -m src.main --config configs/config.yaml
```

The run uses `data/sample/phishing_sample.csv` (10 rows) and writes
`results/results.csv` and `logs/run.log`. To run on the full dataset (Week 7), pass
`--data data/raw/<file>.csv` (the file is not tracked by git).

## Repository Structure

```
phishing-ml-benchmark/
├── configs/
│   └── config.yaml          # all experiment parameters (independent variables, seeds, guardrails)
├── data/
│   └── sample/              # 10-row sample for pipeline smoke tests (full data is git-ignored)
├── docs/
│   └── METHODOLOGY_CARD.md  # Methodology Passport Card (Task 2)
├── notebooks/               # exploratory analysis (checkpoints are git-ignored)
├── results/                 # generated outputs (git-ignored except .gitkeep)
├── src/
│   ├── data.py              # loading + fixed stratified split
│   ├── models.py            # algorithm registry
│   ├── features.py          # feature-selection methods
│   ├── metrics.py           # accuracy/precision/recall/F1, latency, RAM
│   ├── benchmark.py         # algorithm × feature-selection loop, guardrails, baseline delta
│   └── main.py              # CLI entry point
├── Dockerfile
├── environment.yml
├── requirements.txt
└── LICENSE
```

## Output Format

`results/results.csv` has one row per run with: `algorithm`, `feature_selection`,
`n_features_used`, `accuracy`, `precision`, `recall`, `f1`, `train_time_s`, `train_ram_mb`,
`inference_ms_per_sample`, `guardrails_pass`, `f1_delta_vs_baseline`, `status`.

## Reproducibility Notes

- Dependencies are pinned to exact versions; global seed is `experiment.seed` (default 42).
- Feature selectors and scalers are fitted on the training split only (no leakage).
- The sample CSV is **synthetic** and exists only to verify the pipeline; its metrics are meaningless.
- Week 7 will replace the single hold-out split with repeated stratified k-fold and statistical tests.

## License and Citation

Released under the MIT License (see `LICENSE`). If you use this work, please cite:

```
Bedelkhan Alinur, Agadilkhan Nurdaulet. (2026). Machine Learning Approaches for Phishing Website
Detection: A Comparative Performance Analysis. GitHub repository.
```
