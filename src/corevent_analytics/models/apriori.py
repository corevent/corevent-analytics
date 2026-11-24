from __future__ import annotations

from typing import Any

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from corevent_analytics.data.schema import EVENT_CATEGORIES


def transactions_to_onehot(transactions: list[list[str]]) -> pd.DataFrame:
    encoder = TransactionEncoder()
    encoded = encoder.fit(transactions).transform(transactions)
    frame = pd.DataFrame(encoded, columns=encoder.columns_)
    # Ensure stable column order for known categories when present.
    ordered = [c for c in EVENT_CATEGORIES if c in frame.columns]
    extras = [c for c in frame.columns if c not in ordered]
    return frame[ordered + extras]


def run_apriori(
    transactions: list[list[str]],
    *,
    min_support: float = 0.05,
    min_confidence: float = 0.3,
) -> dict[str, Any]:
    if not transactions:
        empty = pd.DataFrame()
        return {
            "frequent_itemsets": empty,
            "rules": empty,
            "n_transactions": 0,
            "min_support": min_support,
            "min_confidence": min_confidence,
        }

    onehot = transactions_to_onehot(transactions)
    frequent = apriori(onehot, min_support=min_support, use_colnames=True)
    if frequent.empty:
        rules = pd.DataFrame()
    else:
        rules = association_rules(
            frequent,
            metric="confidence",
            min_threshold=min_confidence,
        )
        if not rules.empty:
            rules = rules.sort_values(
                by=["lift", "confidence", "support"],
                ascending=False,
            ).reset_index(drop=True)
            rules["antecedents"] = rules["antecedents"].apply(
                lambda items: ", ".join(sorted(items))
            )
            rules["consequents"] = rules["consequents"].apply(
                lambda items: ", ".join(sorted(items))
            )

    return {
        "frequent_itemsets": frequent,
        "rules": rules,
        "n_transactions": len(transactions),
        "min_support": min_support,
        "min_confidence": min_confidence,
    }
