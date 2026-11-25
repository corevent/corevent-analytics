from __future__ import annotations

import logging
from pathlib import Path

from google.cloud import storage

from corevent_analytics.publish.artifacts import API_ARTIFACT_FILES, normalize_gcs_prefix

logger = logging.getLogger(__name__)


def upload_api_artifacts(
    output_dir: Path,
    *,
    bucket_name: str,
    prefix: str = "analytics/",
    client: storage.Client | None = None,
) -> list[str]:
    """Upload API-facing JSON artifacts to GCS. Returns uploaded object paths."""
    bucket = (client or storage.Client()).bucket(bucket_name)
    object_prefix = normalize_gcs_prefix(prefix)
    uploaded: list[str] = []

    for file_name in API_ARTIFACT_FILES:
        local_path = output_dir / file_name
        if not local_path.is_file():
            raise FileNotFoundError(f"Missing analytics artifact: {local_path}")
        object_path = f"{object_prefix}{file_name}"
        bucket.blob(object_path).upload_from_filename(str(local_path), content_type="application/json")
        uploaded.append(object_path)
        logger.info("Uploaded gs://%s/%s", bucket_name, object_path)

    return uploaded
