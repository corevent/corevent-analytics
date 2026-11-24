from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "age",
    "account_age_days",
    "n_orders",
    "total_spent",
    "avg_ticket",
    "n_favorites",
    "n_tickets",
    "checkin_rate",
]


def _select_feature_matrix(user_features: pd.DataFrame) -> pd.DataFrame:
    frame = user_features.copy()
    if "user_id" in frame.columns:
        frame = frame.set_index("user_id")
    cols = [c for c in FEATURE_COLS if c in frame.columns]
    cat_cols = [c for c in frame.columns if c.startswith("cat_")]
    matrix = frame[cols + cat_cols].fillna(0)

    engaged = pd.Series(False, index=frame.index)
    for col in ("n_orders", "n_favorites", "n_tickets"):
        if col in frame.columns:
            engaged = engaged | (frame[col] > 0)
    if engaged.any():
        matrix = matrix.loc[engaged]
    return matrix


def choose_k(
    x_scaled: np.ndarray,
    *,
    k_max: int = 8,
    random_state: int = 42,
) -> dict[str, Any]:
    n_samples = x_scaled.shape[0]
    upper = min(k_max, max(2, n_samples - 1))
    inertias: list[float] = []
    silhouettes: list[float] = []
    ks = list(range(2, upper + 1))

    for k in ks:
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = model.fit_predict(x_scaled)
        inertias.append(float(model.inertia_))
        if n_samples > k and len(set(labels)) > 1:
            silhouettes.append(float(silhouette_score(x_scaled, labels)))
        else:
            silhouettes.append(float("nan"))

    if silhouettes and any(not np.isnan(s) for s in silhouettes):
        best_idx = int(np.nanargmax(silhouettes))
        best_k = ks[best_idx]
    else:
        best_k = ks[0] if ks else 2

    return {
        "ks": ks,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "best_k": best_k,
    }


def run_kmeans(
    user_features: pd.DataFrame,
    *,
    k: int | None = None,
    k_max: int = 8,
    random_state: int = 42,
) -> dict[str, Any]:
    matrix = _select_feature_matrix(user_features)
    if matrix.empty or len(matrix) < 2:
        return {
            "labels": pd.Series(dtype=int),
            "centroids": pd.DataFrame(),
            "metrics": {},
            "selection": {},
            "feature_columns": list(matrix.columns),
            "scaler": None,
            "model": None,
        }

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(matrix)
    selection = choose_k(x_scaled, k_max=k_max, random_state=random_state)
    n_clusters = k or selection["best_k"]
    n_clusters = max(2, min(n_clusters, len(matrix) - 1))

    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = model.fit_predict(x_scaled)

    labeled = matrix.copy()
    labeled["cluster"] = labels
    centroids = labeled.groupby("cluster").mean(numeric_only=True)
    counts = labeled["cluster"].value_counts().sort_index()

    silhouette = (
        float(silhouette_score(x_scaled, labels))
        if len(set(labels)) > 1 and len(matrix) > n_clusters
        else float("nan")
    )

    return {
        "labels": pd.Series(labels, index=matrix.index, name="cluster"),
        "centroids": centroids,
        "cluster_sizes": counts,
        "metrics": {
            "n_clusters": n_clusters,
            "inertia": float(model.inertia_),
            "silhouette": silhouette,
        },
        "selection": selection,
        "feature_columns": list(matrix.columns),
        "scaled_features": x_scaled,
        "model": model,
        "scaler": scaler,
        "matrix": matrix,
    }
