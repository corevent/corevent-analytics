from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from corevent_analytics.publish.artifacts import API_ARTIFACT_FILES, normalize_gcs_prefix
from corevent_analytics.publish.gcs import upload_api_artifacts


def _write_api_artifacts(directory: Path) -> None:
    for name in API_ARTIFACT_FILES:
        (directory / name).write_text("{}", encoding="utf-8")


def test_normalize_gcs_prefix() -> None:
    assert normalize_gcs_prefix("analytics/") == "analytics/"
    assert normalize_gcs_prefix("analytics") == "analytics/"
    assert normalize_gcs_prefix("/analytics/") == "analytics/"
    assert normalize_gcs_prefix("") == ""
    assert normalize_gcs_prefix("/") == ""


def test_upload_api_artifacts_uploads_expected_objects(tmp_path: Path) -> None:
    _write_api_artifacts(tmp_path)
    blob = MagicMock()
    bucket = MagicMock()
    bucket.blob.return_value = blob
    client = MagicMock()
    client.bucket.return_value = bucket

    uploaded = upload_api_artifacts(
        tmp_path,
        bucket_name="corevent",
        prefix="analytics/",
        client=client,
    )

    assert uploaded == [
        "analytics/eda.json",
        "analytics/apriori.json",
        "analytics/kmeans.json",
        "analytics/classification.json",
    ]
    client.bucket.assert_called_once_with("corevent")
    assert bucket.blob.call_count == 4
    assert blob.upload_from_filename.call_count == 4
    first_call = blob.upload_from_filename.call_args_list[0]
    assert first_call.args[0] == str(tmp_path / "eda.json")
    assert first_call.kwargs["content_type"] == "application/json"


def test_upload_api_artifacts_requires_all_files(tmp_path: Path) -> None:
    (tmp_path / "eda.json").write_text("{}", encoding="utf-8")
    client = MagicMock()

    with pytest.raises(FileNotFoundError, match="apriori.json"):
        upload_api_artifacts(tmp_path, bucket_name="corevent", client=client)
