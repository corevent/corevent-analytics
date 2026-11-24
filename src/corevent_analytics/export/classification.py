from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def _model_payload(block: dict[str, Any], *, include_importances: bool = False) -> dict[str, Any]:
    metrics = block.get("metrics") or {}
    payload: dict[str, Any] = {
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "confusionMatrix": metrics.get("confusion_matrix") or [],
        "classificationReport": metrics.get("classification_report") or {},
    }
    if include_importances:
        importances = block.get("feature_importances") or {}
        payload["featureImportances"] = [
            {"feature": str(feature), "importance": float(importance)}
            for feature, importance in importances.items()
        ]
    return payload


def build_classification_payload(
    result: dict[str, Any],
    *,
    source: str,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now(UTC)
    class_balance = result.get("class_balance") or {}
    return {
        "source": source,
        "generatedAt": generated_at.isoformat(),
        "nSamples": int(result.get("n_samples", 0)),
        "nTrain": int(result.get("n_train", 0)) if result.get("n_train") is not None else 0,
        "nTest": int(result.get("n_test", 0)) if result.get("n_test") is not None else 0,
        "classBalance": {str(key): float(value) for key, value in class_balance.items()},
        "featureColumns": list(result.get("feature_columns") or []),
        "models": {
            "randomForest": _model_payload(
                result.get("random_forest") or {},
                include_importances=True,
            ),
            "mlp": _model_payload(result.get("mlp") or {}),
        },
    }
