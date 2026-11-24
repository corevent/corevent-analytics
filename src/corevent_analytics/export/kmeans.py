from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def _centroid_payload(row: pd.Series) -> dict[str, Any]:
    categories: dict[str, float] = {}
    numeric: dict[str, float] = {}
    for key, value in row.items():
        key_str = str(key)
        if key_str == "cluster":
            continue
        number = float(value) if pd.notna(value) else 0.0
        if key_str.startswith("cat_"):
            categories[key_str.removeprefix("cat_")] = number
        else:
            numeric[key_str] = number
    return {**numeric, "categories": categories}


def _build_pca_sample(
    result: dict[str, Any],
    *,
    max_points: int,
    random_state: int,
) -> dict[str, Any]:
    scaled = result.get("scaled_features")
    labels = result.get("labels")
    if scaled is None or labels is None or len(labels) == 0:
        return {"explainedVarianceRatio": [], "points": []}

    n_components = min(2, scaled.shape[1], scaled.shape[0])
    if n_components < 2:
        return {"explainedVarianceRatio": [], "points": []}

    pca = PCA(n_components=2, random_state=random_state)
    projected = pca.fit_transform(scaled)
    points_frame = pd.DataFrame(
        {
            "x": projected[:, 0],
            "y": projected[:, 1],
            "clusterId": labels.to_numpy(),
        }
    )
    if len(points_frame) > max_points:
        points_frame = points_frame.sample(n=max_points, random_state=random_state)

    return {
        "explainedVarianceRatio": [float(value) for value in pca.explained_variance_ratio_],
        "points": [
            {
                "x": float(row["x"]),
                "y": float(row["y"]),
                "clusterId": int(row["clusterId"]),
            }
            for _, row in points_frame.iterrows()
        ],
    }


def build_kmeans_payload(
    result: dict[str, Any],
    *,
    source: str,
    generated_at: datetime | None = None,
    max_pca_points: int = 500,
    random_state: int = 42,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now(UTC)
    metrics = result.get("metrics") or {}
    selection = result.get("selection") or {}
    centroids: pd.DataFrame = result.get("centroids", pd.DataFrame())
    sizes = result.get("cluster_sizes", pd.Series(dtype=int))
    total = int(sizes.sum()) if hasattr(sizes, "sum") and len(sizes) else 0

    candidates: list[dict[str, Any]] = []
    ks = selection.get("ks") or []
    inertias = selection.get("inertias") or []
    silhouettes = selection.get("silhouettes") or []
    for index, k in enumerate(ks):
        silhouette = silhouettes[index] if index < len(silhouettes) else float("nan")
        candidates.append(
            {
                "k": int(k),
                "inertia": float(inertias[index]) if index < len(inertias) else None,
                "silhouette": None if np.isnan(silhouette) else float(silhouette),
            }
        )

    clusters: list[dict[str, Any]] = []
    if not centroids.empty:
        for cluster_id, row in centroids.iterrows():
            size = int(sizes.get(cluster_id, 0)) if hasattr(sizes, "get") else 0
            clusters.append(
                {
                    "clusterId": int(cluster_id),
                    "size": size,
                    "share": float(size / total) if total else 0.0,
                    "centroid": _centroid_payload(row),
                }
            )

    return {
        "source": source,
        "generatedAt": generated_at.isoformat(),
        "metrics": {
            "nClusters": int(metrics.get("n_clusters", 0)) if metrics else 0,
            "inertia": float(metrics["inertia"]) if metrics.get("inertia") is not None else None,
            "silhouette": (
                None
                if metrics.get("silhouette") is None or np.isnan(metrics.get("silhouette"))
                else float(metrics["silhouette"])
            ),
        },
        "kSelection": {
            "bestK": int(selection["best_k"]) if selection.get("best_k") is not None else None,
            "candidates": candidates,
        },
        "clusters": clusters,
        "pcaSample": _build_pca_sample(
            result,
            max_points=max_pca_points,
            random_state=random_state,
        ),
    }
