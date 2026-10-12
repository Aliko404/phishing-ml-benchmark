"""Algorithm registry (independent variable 1)."""
from __future__ import annotations

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

REGISTRY = {
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "svm": SVC,
    "knn": KNeighborsClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "adaboost": AdaBoostClassifier,
}

# Estimators that accept random_state
_SEEDED = {
    "logistic_regression", "decision_tree", "random_forest",
    "svm", "gradient_boosting", "adaboost",
}


def build_model(name: str, params: dict, seed: int):
    if name not in REGISTRY:
        raise KeyError(f"Unknown algorithm '{name}'. Available: {sorted(REGISTRY)}")
    params = dict(params or {})
    if name in _SEEDED:
        params.setdefault("random_state", seed)
    return REGISTRY[name](**params)
