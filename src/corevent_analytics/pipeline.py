from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from corevent_analytics.config import Settings, get_settings
from corevent_analytics.data.load_postgres import load_data
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

logger = logging.getLogger(__name__)


def _ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_safe(value: Any) -> Any:
    return json_safe(value)


def _save_apriori_outputs(result: dict[str, Any], output_dir: Path) -> None:
    rules: pd.DataFrame = result["rules"]
    itemsets: pd.DataFrame = result["frequent_itemsets"]
    if not itemsets.empty:
        export = itemsets.copy()
        export["itemsets"] = export["itemsets"].apply(lambda s: ", ".join(sorted(s)))
        export.to_csv(output_dir / "apriori_itemsets.csv", index=False)
    if not rules.empty:
        rules.to_csv(output_dir / "apriori_rules.csv", index=False)
        top = rules.head(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=top, x="lift", y="antecedents", hue="consequents", ax=ax, dodge=False)
        ax.set_title("Top association rules by lift")
        fig.tight_layout()
        fig.savefig(output_dir / "apriori_top_rules.png", dpi=120)
        plt.close(fig)

    meta = {
        "n_transactions": result["n_transactions"],
        "min_support": result["min_support"],
        "min_confidence": result["min_confidence"],
        "n_itemsets": int(len(itemsets)),
        "n_rules": int(len(rules)),
    }
    (output_dir / "apriori_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )


def _save_kmeans_outputs(result: dict[str, Any], output_dir: Path) -> None:
    if result["centroids"].empty:
        (output_dir / "kmeans_metrics.json").write_text("{}", encoding="utf-8")
        return

    result["centroids"].to_csv(output_dir / "kmeans_centroids.csv")
    sizes = result["cluster_sizes"]
    sizes.to_csv(output_dir / "kmeans_cluster_sizes.csv", header=["size"])

    selection = result.get("selection", {})
    if selection.get("ks"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(selection["ks"], selection["inertias"], marker="o")
        axes[0].set_title("Elbow (inertia)")
        axes[0].set_xlabel("k")
        axes[1].plot(selection["ks"], selection["silhouettes"], marker="o")
        axes[1].set_title("Silhouette")
        axes[1].set_xlabel("k")
        fig.tight_layout()
        fig.savefig(output_dir / "kmeans_k_selection.png", dpi=120)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sizes.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Cluster sizes")
    ax.set_xlabel("cluster")
    ax.set_ylabel("users")
    fig.tight_layout()
    fig.savefig(output_dir / "kmeans_cluster_sizes.png", dpi=120)
    plt.close(fig)

    (output_dir / "kmeans_metrics.json").write_text(
        json.dumps(_json_safe(result["metrics"]), indent=2),
        encoding="utf-8",
    )


def _save_classification_outputs(result: dict[str, Any], output_dir: Path) -> None:
    payload = {
        "n_samples": result.get("n_samples"),
        "n_train": result.get("n_train"),
        "n_test": result.get("n_test"),
        "class_balance": result.get("class_balance"),
        "feature_columns": result.get("feature_columns"),
        "random_forest": result.get("random_forest", {}).get("metrics"),
        "mlp": result.get("mlp", {}).get("metrics"),
        "rf_feature_importances": result.get("random_forest", {}).get("feature_importances"),
    }
    (output_dir / "classification_metrics.json").write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )

    for name in ("random_forest", "mlp"):
        block = result.get(name, {})
        metrics = block.get("metrics")
        if not metrics:
            continue
        cm = metrics.get("confusion_matrix")
        if not cm:
            continue
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"Confusion matrix — {name}")
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        fig.tight_layout()
        fig.savefig(output_dir / f"classification_cm_{name}.png", dpi=120)
        plt.close(fig)

    importances = result.get("random_forest", {}).get("feature_importances")
    if importances:
        top = pd.Series(importances).head(12).sort_values()
        fig, ax = plt.subplots(figsize=(8, 5))
        top.plot(kind="barh", ax=ax, color="teal")
        ax.set_title("Random Forest — top feature importances")
        fig.tight_layout()
        fig.savefig(output_dir / "classification_rf_importances.png", dpi=120)
        plt.close(fig)


def run_pipeline(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    output_dir = _ensure_output_dir(Path(settings.output_dir))
    generated_at = datetime.now(UTC)

    tables, source = load_data(settings)
    logger.info("Data source: %s", source)
    (output_dir / "data_source.json").write_text(
        json.dumps(
            {
                "source": source,
                "counts": {name: int(len(df)) for name, df in tables.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    transactions = build_category_transactions(tables)
    user_features = build_user_features(tables)
    x_cls, y_cls, _ = build_classification_frame(tables)

    user_features.to_csv(output_dir / "user_features.csv", index=False)
    write_json(
        output_dir / "eda.json",
        build_eda_payload(
            tables,
            transactions,
            user_features,
            source=source,
            generated_at=generated_at,
        ),
    )

    apriori_result = run_apriori(
        transactions,
        min_support=settings.min_support,
        min_confidence=settings.min_confidence,
    )
    _save_apriori_outputs(apriori_result, output_dir)
    write_json(
        output_dir / "apriori.json",
        build_apriori_payload(apriori_result, source=source, generated_at=generated_at),
    )

    kmeans_result = run_kmeans(
        user_features,
        k_max=settings.kmeans_k_max,
        random_state=settings.random_state,
    )
    _save_kmeans_outputs(kmeans_result, output_dir)
    write_json(
        output_dir / "kmeans.json",
        build_kmeans_payload(
            kmeans_result,
            source=source,
            generated_at=generated_at,
            random_state=settings.random_state,
        ),
    )

    classify_result = run_classification(
        x_cls,
        y_cls,
        test_size=settings.test_size,
        random_state=settings.random_state,
    )
    _save_classification_outputs(classify_result, output_dir)
    write_json(
        output_dir / "classification.json",
        build_classification_payload(
            classify_result,
            source=source,
            generated_at=generated_at,
        ),
    )

    summary = {
        "source": source,
        "n_transactions": apriori_result["n_transactions"],
        "n_rules": int(len(apriori_result["rules"])),
        "kmeans": kmeans_result.get("metrics", {}),
        "classification_samples": classify_result.get("n_samples"),
        "rf_f1": (classify_result.get("random_forest") or {}).get("metrics", {}).get("f1"),
        "mlp_f1": (classify_result.get("mlp") or {}).get("metrics", {}).get("f1"),
        "output_dir": str(output_dir),
    }
    (output_dir / "pipeline_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2),
        encoding="utf-8",
    )
    logger.info("Pipeline finished: %s", summary)
    return {
        "tables": tables,
        "source": source,
        "transactions": transactions,
        "user_features": user_features,
        "apriori": apriori_result,
        "kmeans": kmeans_result,
        "classification": classify_result,
        "summary": summary,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_pipeline()


if __name__ == "__main__":
    main()
