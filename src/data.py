"""Strict numeric dataset validation and deterministic stratified split."""
from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split

def load_split(cfg: dict, seed: int):
    d = cfg["data"]
    df = pd.read_csv(d["path"])
    target = d["target_column"]
    if target not in df.columns:
        raise ValueError(f"Missing target '{target}' in {d['path']}")
    if df.empty or df.isna().any().any():
        raise ValueError("Dataset is empty or contains missing values; clean data first")
    X, y = df.drop(columns=[target]), df[target]
    if not all(pd.api.types.is_numeric_dtype(dt) for dt in X.dtypes):
        raise ValueError("Expected numeric features; encode categorical columns before benchmarking")
    if not set(y.unique()).issubset({0,1}) or y.nunique()!=2:
        raise ValueError("Expected binary target labels 0=legitimate, 1=phishing")
    if y.value_counts().min()<2:
        raise ValueError("Each class needs at least 2 samples for train/test splitting")
    stratify = y if d.get('stratify', True) else None
    return train_test_split(X, y, test_size=d['test_size'], random_state=seed, stratify=stratify)
