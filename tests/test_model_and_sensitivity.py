
import pandas as pd

from src.models.predict_review_rating import evaluate
from src.models.analyze_ranking_sensitivity import rank_products


def test_evaluate_returns_expected_metrics():
    y_true = [1, 2, 3, 4, 5, 5]
    y_pred = [1, 2, 3, 4, 5, 4]

    metrics = evaluate(y_true, y_pred)

    assert metrics["accuracy"] == 0.8333
    assert 0 <= metrics["balanced_accuracy"] <= 1
    assert 0 <= metrics["macro_f1"] <= 1
    assert 0 <= metrics["weighted_f1"] <= 1


def test_ranking_sensitivity_returns_sorted_products():
    reviews = pd.DataFrame(
        {
            "parent_asin": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "rating": [5, 5, 5, 4, 4, 4, 3, 3, 3],
        }
    )

    ranking = rank_products(reviews, prior_strength=5)

    assert len(ranking) == 3
    assert ranking["rank"].tolist() == [1, 2, 3]
    assert ranking["adjusted_score"].is_monotonic_decreasing
    assert ranking["parent_asin"].is_unique


def test_ranking_sensitivity_rejects_invalid_prior_strength():
    reviews = pd.DataFrame(
        {
            "parent_asin": ["A", "A", "A"],
            "rating": [5, 4, 5],
        }
    )

    # The current implementation accepts zero, so this test documents
    # the valid positive-parameter contract we want to enforce.
    try:
        rank_products(reviews, prior_strength=0)
    except ValueError:
        return

    raise AssertionError("Expected ValueError for prior_strength=0")