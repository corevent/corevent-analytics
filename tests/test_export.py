from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from corevent_analytics.data.generate_synthetic import generate_synthetic_dataset
from corevent_analytics.data.preprocess import (
    build_category_transactions,
    build_classification_frame,
    build_user_features,
)
from corevent_analytics.export import (
    build_apriori_payload,
    build_classification_payload,
    build_eda_payload,
    build_kmeans_payload,
    json_safe,
    write_json,
)
from corevent_analytics.models.apriori import run_apriori
from corevent_analytics.models.classify import run_classification
from corevent_analytics.models.kmeans import run_kmeans


FIXED_AT = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)


def _small_tables() -> dict[str, pd.DataFrame]:
    return generate_synthetic_dataset(n_users=80, n_events=20, seed=7)


def test_json_safe_handles_nested_numpy_and_datetime() -> None:
    payload = json_safe(
        {
            "when": FIXED_AT,
            "values": [pd.Series([1]).iloc[0]],
            "nested": {"ok": True},
        }
    )
    assert payload["when"] == FIXED_AT.isoformat()
    assert payload["values"] == [1]
    assert payload["nested"]["ok"] is True
    json.dumps(payload)


def test_eda_payload_shape() -> None:
    tables = _small_tables()
    transactions = build_category_transactions(tables)
    user_features = build_user_features(tables)
    payload = build_eda_payload(
        tables,
        transactions,
        user_features,
        source="synthetic",
        generated_at=FIXED_AT,
    )
    assert payload["source"] == "synthetic"
    assert payload["generatedAt"] == FIXED_AT.isoformat()
    assert "users" in payload["tableCounts"]
    assert isinstance(payload["eventsByCategory"], list)
    assert isinstance(payload["ordersByStatus"], list)
    assert payload["transactionsSummary"]["nTransactions"] == len(transactions)
    assert "age" in payload["userFeatureStats"]
    json.dumps(json_safe(payload))


def test_apriori_payload_shape() -> None:
    tables = _small_tables()
    transactions = build_category_transactions(tables)
    result = run_apriori(transactions, min_support=0.05, min_confidence=0.2)
    payload = build_apriori_payload(result, source="synthetic", generated_at=FIXED_AT)
    assert payload["meta"]["nTransactions"] == len(transactions)
    assert isinstance(payload["frequentItemsets"], list)
    assert isinstance(payload["rules"], list)
    if payload["rules"]:
        assert isinstance(payload["rules"][0]["antecedents"], list)
        assert isinstance(payload["rules"][0]["consequents"], list)
    json.dumps(json_safe(payload))


def test_kmeans_payload_shape_and_pca_sample_cap() -> None:
    tables = _small_tables()
    user_features = build_user_features(tables)
    result = run_kmeans(user_features, k_max=4, random_state=7)
    payload = build_kmeans_payload(
        result,
        source="synthetic",
        generated_at=FIXED_AT,
        max_pca_points=25,
        random_state=7,
    )
    assert "metrics" in payload
    assert "kSelection" in payload
    assert isinstance(payload["clusters"], list)
    assert len(payload["pcaSample"]["points"]) <= 25
    if payload["pcaSample"]["points"]:
        point = payload["pcaSample"]["points"][0]
        assert set(point.keys()) == {"x", "y", "clusterId"}
        assert "userId" not in point
    json.dumps(json_safe(payload))


def test_classification_payload_shape() -> None:
    tables = _small_tables()
    x, y, _ = build_classification_frame(tables)
    result = run_classification(x, y, test_size=0.3, random_state=7)
    payload = build_classification_payload(result, source="synthetic", generated_at=FIXED_AT)
    assert payload["nSamples"] == len(x)
    assert "randomForest" in payload["models"]
    assert "mlp" in payload["models"]
    assert isinstance(payload["models"]["randomForest"]["featureImportances"], list)
    json.dumps(json_safe(payload))


def test_write_json_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "eda.json"
    write_json(path, {"source": "synthetic", "generatedAt": FIXED_AT.isoformat(), "ok": True})
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["ok"] is True
    assert loaded["source"] == "synthetic"
