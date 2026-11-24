from corevent_analytics.data.generate_synthetic import generate_synthetic_dataset
from corevent_analytics.data.preprocess import build_user_features
from corevent_analytics.models.kmeans import run_kmeans


def test_kmeans_returns_expected_clusters() -> None:
    tables = generate_synthetic_dataset(n_users=200, n_events=60, seed=5)
    features = build_user_features(tables)
    result = run_kmeans(features, k=3, k_max=6, random_state=42)
    assert result["metrics"]["n_clusters"] == 3
    assert len(result["labels"]) > 0
    assert set(result["labels"].unique()) <= {0, 1, 2}
    assert len(result["centroids"]) == 3
