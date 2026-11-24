from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def _safe_split(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            zero_division=0,
            output_dict=True,
        ),
    }


def run_classification(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict[str, Any]:
    if x.empty or y.empty or y.nunique() < 2:
        return {
            "random_forest": {},
            "mlp": {},
            "feature_columns": list(x.columns),
            "n_samples": int(len(x)),
            "class_balance": y.value_counts(normalize=True).to_dict() if len(y) else {},
        }

    x_train, x_test, y_train, y_test = _safe_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(x_train, y_train)
    rf_pred = rf.predict(x_test)
    rf_importances = (
        pd.Series(rf.feature_importances_, index=x.columns)
        .sort_values(ascending=False)
        .to_dict()
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=500,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.15,
    )
    mlp.fit(x_train_scaled, y_train)
    mlp_pred = mlp.predict(x_test_scaled)

    return {
        "random_forest": {
            "metrics": _metrics(y_test.to_numpy(), rf_pred),
            "feature_importances": rf_importances,
            "model": rf,
            "y_test": y_test,
            "y_pred": rf_pred,
        },
        "mlp": {
            "metrics": _metrics(y_test.to_numpy(), mlp_pred),
            "model": mlp,
            "scaler": scaler,
            "y_test": y_test,
            "y_pred": mlp_pred,
        },
        "feature_columns": list(x.columns),
        "n_samples": int(len(x)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "class_balance": y.value_counts(normalize=True).to_dict(),
    }
