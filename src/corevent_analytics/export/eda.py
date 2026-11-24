from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any

import pandas as pd


def _value_counts(series: pd.Series) -> list[dict[str, Any]]:
    if series.empty:
        return []
    counts = series.value_counts()
    return [{"name": str(name), "count": int(count)} for name, count in counts.items()]


def _feature_stats(frame: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for column in columns:
        if column not in frame.columns:
            continue
        described = frame[column].describe()
        stats[column] = {
            "count": float(described.get("count", 0)),
            "mean": float(described.get("mean", 0)),
            "std": float(described.get("std", 0)) if pd.notna(described.get("std")) else 0.0,
            "min": float(described.get("min", 0)),
            "p25": float(described.get("25%", 0)),
            "p50": float(described.get("50%", 0)),
            "p75": float(described.get("75%", 0)),
            "max": float(described.get("max", 0)),
        }
    return stats


def build_eda_payload(
    tables: dict[str, pd.DataFrame],
    transactions: list[list[str]],
    user_features: pd.DataFrame,
    *,
    source: str,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now(UTC)
    category_counts = Counter(category for transaction in transactions for category in transaction)
    avg_categories = (
        sum(len(transaction) for transaction in transactions) / len(transactions)
        if transactions
        else 0.0
    )

    events = tables.get("events", pd.DataFrame())
    orders = tables.get("orders", pd.DataFrame())

    return {
        "source": source,
        "generatedAt": generated_at.isoformat(),
        "tableCounts": {name: int(len(frame)) for name, frame in tables.items()},
        "eventsByCategory": (
            [{"category": item["name"], "count": item["count"]} for item in _value_counts(events["category"])]
            if not events.empty and "category" in events.columns
            else []
        ),
        "ordersByStatus": (
            [{"status": item["name"], "count": item["count"]} for item in _value_counts(orders["status"])]
            if not orders.empty and "status" in orders.columns
            else []
        ),
        "transactionsSummary": {
            "nTransactions": len(transactions),
            "avgCategoriesPerUser": float(avg_categories),
            "categoryUserCounts": [
                {"category": category, "users": int(count)}
                for category, count in sorted(category_counts.items(), key=lambda item: (-item[1], item[0]))
            ],
        },
        "userFeatureStats": _feature_stats(
            user_features,
            ["age", "n_orders", "n_favorites", "avg_ticket", "checkin_rate"],
        ),
    }
