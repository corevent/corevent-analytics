from corevent_analytics.data.generate_synthetic import generate_synthetic_dataset
from corevent_analytics.data.preprocess import (
    build_category_transactions,
    build_classification_frame,
    build_user_features,
)
from corevent_analytics.data.schema import EVENT_CATEGORIES


def test_build_category_transactions_format() -> None:
    tables = generate_synthetic_dataset(n_users=100, n_events=40, seed=2)
    transactions = build_category_transactions(tables)
    assert len(transactions) > 0
    for txn in transactions:
        assert isinstance(txn, list)
        assert len(txn) == len(set(txn))
        assert set(txn).issubset(set(EVENT_CATEGORIES))


def test_build_user_features_columns() -> None:
    tables = generate_synthetic_dataset(n_users=100, n_events=40, seed=3)
    features = build_user_features(tables)
    assert "user_id" in features.columns
    assert "age" in features.columns
    assert "n_favorites" in features.columns
    for cat in EVENT_CATEGORIES:
        assert f"cat_{cat}" in features.columns
    assert len(features) == len(tables["users"])


def test_classification_frame_excludes_non_favorites() -> None:
    tables = generate_synthetic_dataset(n_users=150, n_events=50, seed=4)
    x, y, cols = build_classification_frame(tables)
    assert len(x) == len(y)
    assert len(x) > 0
    assert set(y.unique()).issubset({0, 1})
    assert "n_orders" not in cols
    assert "total_spent" not in cols
    assert y.nunique() == 2
