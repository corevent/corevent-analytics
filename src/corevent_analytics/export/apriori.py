from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd


def _split_items(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (set, frozenset, list, tuple)):
        return sorted(str(item) for item in value)
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def build_apriori_payload(
    result: dict[str, Any],
    *,
    source: str,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now(UTC)
    itemsets: pd.DataFrame = result.get("frequent_itemsets", pd.DataFrame())
    rules: pd.DataFrame = result.get("rules", pd.DataFrame())

    frequent_itemsets: list[dict[str, Any]] = []
    if not itemsets.empty:
        for _, row in itemsets.iterrows():
            frequent_itemsets.append(
                {
                    "items": _split_items(row.get("itemsets")),
                    "support": float(row.get("support", 0)),
                }
            )

    exported_rules: list[dict[str, Any]] = []
    if not rules.empty:
        for _, row in rules.iterrows():
            exported_rules.append(
                {
                    "antecedents": _split_items(row.get("antecedents")),
                    "consequents": _split_items(row.get("consequents")),
                    "antecedentSupport": float(row.get("antecedent support", 0)),
                    "consequentSupport": float(row.get("consequent support", 0)),
                    "support": float(row.get("support", 0)),
                    "confidence": float(row.get("confidence", 0)),
                    "lift": float(row.get("lift", 0)),
                    "leverage": float(row.get("leverage", 0)) if "leverage" in rules.columns else None,
                    "conviction": float(row.get("conviction", 0)) if "conviction" in rules.columns else None,
                    "zhangsMetric": (
                        float(row.get("zhangs_metric", 0)) if "zhangs_metric" in rules.columns else None
                    ),
                    "jaccard": float(row.get("jaccard", 0)) if "jaccard" in rules.columns else None,
                    "certainty": float(row.get("certainty", 0)) if "certainty" in rules.columns else None,
                    "kulczynski": float(row.get("kulczynski", 0)) if "kulczynski" in rules.columns else None,
                }
            )

    return {
        "source": source,
        "generatedAt": generated_at.isoformat(),
        "meta": {
            "nTransactions": int(result.get("n_transactions", 0)),
            "minSupport": float(result.get("min_support", 0)),
            "minConfidence": float(result.get("min_confidence", 0)),
            "nItemsets": int(len(itemsets)),
            "nRules": int(len(rules)),
        },
        "frequentItemsets": frequent_itemsets,
        "rules": exported_rules,
    }
