"""Dependent variables: quality, latency and memory metrics."""
from __future__ import annotations

import time
import tracemalloc

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def quality_metrics(y_true, y_pred) -> dict:
    kw = {"zero_division": 0}
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, **kw),
        "recall": recall_score(y_true, y_pred, **kw),
        "f1": f1_score(y_true, y_pred, **kw),
    }


def timed_fit(pipeline, X, y) -> tuple[float, float]:
    """Return (train_time_s, peak_ram_mb) for fitting the pipeline."""
    tracemalloc.start()
    t0 = time.perf_counter()
    pipeline.fit(X, y)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak / (1024 * 1024)


def inference_ms_per_sample(pipeline, X, repeats: int, warmup: int) -> float:
    """Median per-sample prediction latency in milliseconds."""
    for _ in range(warmup):
        pipeline.predict(X)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        pipeline.predict(X)
        times.append(time.perf_counter() - t0)
    return float(np.median(times) / len(X) * 1000.0)
