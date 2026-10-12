"""Data loading and the fixed train/test split (controlled variable)."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def load_split(cfg: dict, seed: int):
    """Load the CSV and return X_train, X_test, y_train, y_test.

    The split is deterministic (seed from config) so every algorithm and
    every feature-selection method sees exactly the same data.
    """
    d = cfg["data"]
    df = pd.read_csv(d["path"])
    target = d["target_column"]
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {d['path']}")
    X = df.drop(columns=[target])
    y = df[target]
    stratify = y if d.get("stratify", True) else None
    return train_test_split(
        X, y, test_size=d["test_size"], random_state=seed, stratify=stratify
    )
