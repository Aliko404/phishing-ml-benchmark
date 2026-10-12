"""Benchmark loop: every algorithm x every feature-selection method."""
from __future__ import annotations

import logging
import signal
import time

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_split
from .features import build_selector
from .metrics import inference_ms_per_sample, quality_metrics, timed_fit
from .models import build_model

log = logging.getLogger("benchmark")


class RunTimeout(Exception):
    pass


def _alarm(_signum, _frame):
    raise RunTimeout()


def run_benchmark(cfg: dict) -> pd.DataFrame:
    seed = cfg["experiment"]["seed"]
    meas = cfg["measurement"]
    X_train, X_test, y_train, y_test = load_split(cfg, seed)
    n_features = X_train.shape[1]
    log.info("Data: train=%d test=%d features=%d", len(X_train), len(X_test), n_features)

    rows = []
    for fs_spec in cfg["feature_selection"]:
        for algo, params in cfg["algorithms"].items():
            steps = [("scale", StandardScaler())]
            selector = build_selector(fs_spec, n_features, seed)
            if selector is not None:
                steps.append(("select", selector))
            steps.append(("clf", build_model(algo, params, seed)))
            pipe = Pipeline(steps)

            row = {"algorithm": algo, "feature_selection": fs_spec["method"],
                   "n_features_used": n_features, "status": "ok"}
            try:
                if hasattr(signal, "SIGALRM"):
                    signal.signal(signal.SIGALRM, _alarm)
                    signal.alarm(int(meas["timeout_seconds"]))
                train_s, ram_mb = timed_fit(pipe, X_train, y_train)
                y_pred = pipe.predict(X_test)
                row.update(quality_metrics(y_test, y_pred))
                row["train_time_s"] = train_s
                row["train_ram_mb"] = ram_mb
                row["inference_ms_per_sample"] = inference_ms_per_sample(
                    pipe, X_test, meas["inference_repeats"], meas["warmup_runs"]
                )
                if "select" in pipe.named_steps:
                    sel = pipe.named_steps["select"]
                    row["n_features_used"] = (
                        sel.n_components_ if hasattr(sel, "n_components_")
                        else int(sel.get_support().sum())
                    )
            except RunTimeout:
                row["status"] = "timeout"
            except Exception as exc:  # keep the benchmark running
                log.exception("Run failed: %s / %s", algo, fs_spec["method"])
                row["status"] = f"error: {exc}"
            finally:
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(0)
            log.info("%s | %s | f1=%s", algo, fs_spec["method"], row.get("f1"))
            rows.append(row)

    df = pd.DataFrame(rows)
    return add_guardrails(df, cfg)


def add_guardrails(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    g = cfg["guardrails"]
    ok = df["status"] == "ok"
    df["guardrails_pass"] = (
        ok
        & (df["recall"] >= g["min_recall"])
        & (df["inference_ms_per_sample"] <= g["max_inference_ms_per_sample"])
        & (df["train_ram_mb"] <= g["max_train_ram_mb"])
    )
    # delta of the primary metric (F1) against the baseline run
    b = cfg["baseline"]
    base = df[(df.algorithm == b["algorithm"]) & (df.feature_selection == b["feature_selection"])]
    if len(base) and base["status"].iloc[0] == "ok":
        df["f1_delta_vs_baseline"] = df["f1"] - base["f1"].iloc[0]
    return df
