"""Feature-selection methods (independent variable 2).

Each builder returns an unfitted sklearn transformer, or None for 'none'.
Selectors are fitted on the training split only (no test leakage).
"""
from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif


def build_selector(spec: dict, n_features: int, seed: int):
    method = spec["method"]
    if method == "none":
        return None
    if method == "mutual_info":
        k = min(spec.get("k", 5), n_features)
        return SelectKBest(
            lambda X, y: mutual_info_classif(X, y, random_state=seed, n_neighbors=2),
            k=k,
        )
    if method == "rf_importance":
        k = min(spec.get("k", 5), n_features)
        rf = RandomForestClassifier(n_estimators=50, random_state=seed)
        return SelectFromModel(rf, max_features=k, threshold=-float("inf"))
    if method == "pca":
        n = min(spec.get("n_components", 5), n_features)
        return PCA(n_components=n, random_state=seed)
    raise ValueError(f"Unknown feature-selection method '{method}'")
