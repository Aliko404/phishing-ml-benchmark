# Task 3 — ML Domain Model and Mathematical Formalization

**Research track:** Machine Learning / AI. **Task:** binary phishing-site classification with measurable prediction latency. This document formalizes the evaluated pipeline rather than claiming every classifier optimizes the same loss.

## 1. System and feature pipeline

`Labeled feature table → stratified train/test split → scaler fitted on train only → classifier → prediction → timing collector → JSONL → analysis → figures`

- Input feature vector: $\mathbf{x}_i\in\mathbb{R}^{d}$.
- Target: $y_i\in\{0,1\}$, with 1 = phishing and 0 = benign.
- Predictor: $\hat{y}_i=f_{\theta}(\mathbf{x}_i)$; probability estimate $p_i=P_{\theta}(y_i=1\mid\mathbf{x}_i)$ where available.
- The small sample is for software verification only. It must not be used to claim real-world detection quality.

## 2. Mathematical objective (logistic-regression baseline)

The logistic regression baseline minimizes regularized binary cross-entropy (log loss):

$$
\mathcal L(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]+\lambda\lVert\mathbf w\rVert_2^2.
$$

The exact scaling/regularization convention depends on the scikit-learn solver and `C`. **Random Forest does not minimize this differentiable objective**; it constructs decision trees using impurity-based split criteria. Thus, this loss is a formalization of the logistic-regression baseline only, not a claim about every model.

## 3. Prediction quality and performance metrics

With phishing as the positive class:

$$\mathrm{Precision}=\frac{TP}{TP+FP},\quad \mathrm{Recall}=\frac{TP}{TP+FN},\quad F_1=\frac{2PR}{P+R}. $$

For inference call $j$, $t_j=1000\cdot (t_{\mathrm{end},j}-t_{\mathrm{start},j})$ ms measured around `predict()` with `time.perf_counter()`; `batch_size` denotes *number of predictions in a call*, **not** actual requests per second.

$$\widetilde t=\mathrm{median}\{t_j\},\quad p_{95}=Q_{0.95}(t),\quad p_{99}=Q_{0.99}(t).$$

A percentile bootstrap CI for the median is computed via $B=2000$ samples drawn **with replacement** from successful benchmark latencies in each (model, batch size) group using `numpy.random.default_rng(42)`. The confidence interval is $[Q_{0.025}(\widetilde t^*),Q_{0.975}(\widetilde t^*)]$. This interval captures sample uncertainty under an iid assumption and does not automatically account for correlated measurements.

$$\mathrm{ErrorRate}(\%)=100\frac{n_{\mathrm{failed}}}{n_{\mathrm{benchmark}}}.$$

Warm-up observations are discarded **before** summary calculations; failed prediction events remain in the denominator of the error-rate guardrail. Error latency is not silently treated as valid successful prediction latency.

## 4. Notation table

| Symbol | Meaning | Unit |
|:--|:--|:--|
| $N$ | Number of labeled training samples | samples |
| $d$ | Input feature dimension | features |
| $y_i$ | Ground-truth binary label | dimensionless |
| $\mathbf x_i$ | Feature vector | varies by feature |
| $\theta,\mathbf w$ | Model parameters, logistic weights | model-dependent |
| $p_i$ | Estimated phishing probability | [0,1] |
| $\lambda$ | Logistic L2 regularization strength | model-dependent |
| $TP, FP, FN$ | Classification counts | samples |
| $P,R,F_1$ | Precision, recall and F1 | [0,1] |
| $t_j$ | One `predict()` wall-clock latency | ms |
| $\widetilde t$ | Median prediction latency | ms |
| $Q_q$ | Empirical quantile of order $q$ | ms for latency |
| $B$ | Number of bootstrap resamples | resamples |
| $n_{\mathrm{failed}}$ | Failed benchmark calls | calls |
| $n_{\mathrm{benchmark}}$ | All benchmark calls after warm-up | calls |

## 5. Variables and threats to validity

- **Independent:** classifier (LogisticRegression / RandomForest); inference batch size (1, 2, 4, 6, 9).
- **Dependent:** prediction latency median, p95/p99, 95% CI and error rate; F1/recall in the separate ML-classification benchmark.
- **Controlled:** common dataset split, seed 42, same hardware/session and preprocessing protocol.
- **Threats:** 10-row fixture, small train/test split, non-independent consecutive timings, OS background load, no throughput/load generator, and mixed batch sizes in distribution plot. Outputs are illustrative *smoke-test artefacts*, not publishable generalization claims.
