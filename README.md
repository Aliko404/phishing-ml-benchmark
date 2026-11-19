# Phishing Website Detection — Machine Learning Benchmark

**Status:** Reproducible measurement pipeline with recorded UCI inference telemetry. Measurements depend on the machine used and should be regenerated on your Linux system.

## Dataset and Measurement Protocol
- **Input:** UCI Machine Learning Repository, *Phishing Websites*, dataset **327**, 11,055 real labeled website examples, 30 numeric features; DOI: https://doi.org/10.24432/C51W2X.
- **Telemetry:** Each JSONL entry will be a **real elapsed-time measurement** of an actual trained classifier `predict()` call on those data, collected locally on Linux with `time.perf_counter()`. The publicly available input CSV is **not** a set of network traffic logs.
- Label mapping: UCI `-1` = phishing => project `1`; UCI `+1` = legitimate => project `0`.
- The 10-row CSV in `data/sample/` is solely a smoke-test fixture, not research data. Previous toy logs and plots are excluded from this release.

## Linux Quickstart — collect actual measurements
Use Python 3.11 (required by the pinned environment). Run all commands at repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.download_uci
python -m src.collect_inference --data data/external/uci327_phishing_websites.csv --out data/raw/test_run.jsonl --runs 120
python -m src.process_data --input data/raw/test_run.jsonl
python -m src.visualize --input data/raw/test_run.jsonl
jupyter lab notebooks/01_data_analysis.ipynb
```

In Jupyter use **Kernel → Restart & Run All**, then save to preserve outputs. On a fresh GitHub clone without `data/raw/test_run.jsonl`, the notebook uses the committed 130-call sample **only as a demonstration** and saves outputs to separate sample-preview paths. Run `bash run_dataset.sh` first for the complete dataset-based experiment.

`python -m src.download_uci` requires internet **only on first download**. The GitHub repository does not include the dataset CSV. Download it from UCI using `python -m src.download_uci`. Dataset provenance and checksums are documented in the repository. For data integrity verification, examine `data/external/uci327_provenance.json` (local dataset SHA256) and `dataset_sha256` in every JSONL record.

## Experimental Methodology
A fixed stratified train/test split (`random_state=42`) trains Logistic Regression (baseline) and Random Forest. For each model and batch size (1, 2, 4, 6, 9), `--runs 120` runs 12 warm-up calls followed by 108 benchmark calls (per condition); elapsed milliseconds are measured from actual `predict()` operations. Logging happens after timing. This is a **CPU inference benchmark**, not real website HTTP traffic, true production throughput or RPS.

`src/process_data.py` validates the log schema, excludes `phase='warmup'`, reports per-model error rate and median, p25/p75, p95/p99, and 2,000 seeded bootstrap resamples for the 95% CI of the median. The bootstrap characterizes repeat-call variability on this setup, not uncertainty across websites or hardware. `src/visualize.py` creates two figures (PDF and 300-DPI PNG) in `reports/figures/`; axes have units, and the latency chart shows uncertainty bars. `docs/domain_model.md` provides ML mathematical notation.

## Experiment artifacts
- `data/raw/test_run.jsonl` — full local measurements (ignored by Git); `data/raw/test_run_sample.jsonl` — 130 authentic measured records spanning all model/batch combinations (committed, for schema inspection only)
- `data/processed/metrics_summary.csv` — saved summary of the full local run (committed; reproduce by running the experiment on Linux)
- `reports/figures/fig1_latency_vs_load.pdf` and `.png`
- `reports/figures/fig2_distribution_boxplot.pdf` and `.png`
- `notebooks/01_data_analysis.ipynb` — rerunnable; if the full raw log is absent, the committed 130-call sample is used strictly as a preview without replacing published full-run CSV or figures

### Results & Discussion
The archive includes measured inference telemetry and a grouped summary in [`data/processed/metrics_summary.csv`](data/processed/metrics_summary.csv). These measurements were recorded on the machine that produced the logs; regenerate them on your own Linux machine for reproducible hardware-specific results. The figures below summarize those recorded observations. No claim about production HTTP traffic or general model superiority follows from these measurements.

### Recorded inference comparison (saved full-run summary)

The following figures come from the committed `metrics_summary.csv` (108 successful post-warm-up calls per model/batch condition). They are **machine-specific historical measurements**, not results recomputed from the 130-row sample committed to Git. A fresh Linux run will produce different absolute timings.

| Batch size | Logistic Regression median (ms) [95% bootstrap CI] | Random Forest median (ms) [95% bootstrap CI] |
|:--|--:|--:|
| 1 | 2.187 [2.172, 2.199] | 13.980 [13.939, 14.045] |
| 2 | 1.736 [1.721, 1.749] | 14.974 [14.657, 15.082] |
| 4 | 1.708 [1.702, 1.719] | 14.175 [14.139, 14.238] |
| 6 | 1.720 [1.707, 1.728] | 14.297 [14.214, 14.389] |
| 9 | 1.706 [1.689, 1.726] | 14.702 [14.566, 15.162] |

Both variants reported 0% failed calls in this saved run. These inference-timing numbers do **not** establish model detection quality.

![Inference latency versus batch size](reports/figures/fig1_latency_vs_load.png)

![Inference latency distribution](reports/figures/fig2_distribution_boxplot.png)

Comparing only inference latency does **not** establish phishing detection accuracy; run the separate `src.main` 7×4 benchmark for classification metrics after adapting the dataset configuration.

### Integrity / limitations
The UCI dataset contains derived URL/site attributes, not request arrival times or HTTP status. HTTP-like `status_code` in our JSONL is a **local success/failure marker** (`200` for successful predict, `500` for exception), not a true response from a web server. Measured runtime depends on hardware and concurrent load. Batch size is not requests per second. For a full research claim, assess classification F1/recall with held-out data and repeated cross-validation and discuss dataset provenance / representativeness.

## Repository Structure and Data Management
Large raw logs and dataset CSVs, temporary logs, and `.venv/` are excluded by `.gitignore`; the small real-measurement sample and summary CSV are tracked for review. Review generated figures and notebook before committing them. Old committed dumps must be untracked (`git rm --cached data/raw/sample_test_run.jsonl`) and, if necessary, removed from history with coordinated `git filter-repo`.

## Reproducibility and Citation
Source: [UCI 327 Phishing Websites](https://archive.ics.uci.edu/dataset/327/phishing+websites), DOI 10.24432/C51W2X. See [domain model](docs/domain_model.md) and [data provenance](docs/REAL_DATA_PROVENANCE.md). Source code released under the included MIT LICENSE; source dataset retains its own terms.

### GitHub check
The repository includes a small sample of real telemetry (`data/raw/test_run_sample.jsonl`) to inspect schema and warm-up flags. This sample is **not** the full input that produced `metrics_summary.csv`; reproduce the full run using `bash run_dataset.sh`. The checked-in summary and figures are machine-dependent records from the original measurement environment.
