"""API-facing artifact filenames (must match api-corevent)."""

from __future__ import annotations

API_ARTIFACT_FILES: tuple[str, ...] = (
    "eda.json",
    "apriori.json",
    "kmeans.json",
    "classification.json",
)


def normalize_gcs_prefix(prefix: str) -> str:
    stripped = prefix.strip("/")
    if not stripped:
        return ""
    return f"{stripped}/"
