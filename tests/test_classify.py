from corevent_analytics.data.generate_synthetic import generate_synthetic_dataset
from corevent_analytics.data.preprocess import build_classification_frame
from corevent_analytics.models.classify import run_classification


def test_classification_metrics_shape() -> None:
    tables = generate_synthetic_dataset(n_users=250, n_events=70, seed=6)
    x, y, _ = build_classification_frame(tables)
    result = run_classification(x, y, test_size=0.25, random_state=42)
    assert result["n_samples"] == len(x)
    for model_name in ("random_forest", "mlp"):
        metrics = result[model_name]["metrics"]
        for key in ("accuracy", "precision", "recall", "f1"):
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0
        cm = metrics["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2
