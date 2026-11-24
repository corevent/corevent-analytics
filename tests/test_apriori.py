from corevent_analytics.models.apriori import run_apriori


def test_apriori_finds_rules_on_deterministic_transactions() -> None:
    transactions = [
        ["music", "gastronomy"],
        ["music", "gastronomy", "art_culture"],
        ["music", "gastronomy"],
        ["tech", "business"],
        ["tech", "business", "education"],
        ["music", "gastronomy", "fashion_beauty"],
        ["sports", "health_wellness"],
        ["music", "gastronomy"],
        ["tech", "business"],
        ["music", "art_culture", "gastronomy"],
    ]
    result = run_apriori(transactions, min_support=0.3, min_confidence=0.5)
    assert result["n_transactions"] == 10
    assert not result["frequent_itemsets"].empty
    assert not result["rules"].empty
    rules_text = (
        result["rules"]["antecedents"] + "->" + result["rules"]["consequents"]
    ).tolist()
    assert any("music" in r and "gastronomy" in r for r in rules_text)


def test_apriori_empty_input() -> None:
    result = run_apriori([], min_support=0.1, min_confidence=0.5)
    assert result["n_transactions"] == 0
    assert result["rules"].empty
