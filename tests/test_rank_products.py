import pandas as pd

from src.models.rank_products import build_product_ranking


def test_ranking_excludes_products_with_too_few_reviews():
    reviews = pd.DataFrame(
        {
            "parent_asin": ["A", "A", "B", "C", "C", "C"],
            "rating": [5, 5, 5, 4, 4, 4],
            "helpful_vote": [0, 1, 0, 0, 0, 0],
            "is_verified_purchase": [True, True, True, True, True, True],
        }
    )

    ranking = build_product_ranking(reviews)

    assert "B" not in ranking["parent_asin"].values
    assert "A" not in ranking["parent_asin"].values
    assert "C" in ranking["parent_asin"].values


def test_ranking_scores_are_between_one_and_five():
    reviews = pd.DataFrame(
        {
            "parent_asin": ["A", "A", "A", "B", "B", "B"],
            "rating": [5, 5, 4, 3, 4, 4],
            "helpful_vote": [0, 1, 0, 0, 0, 0],
            "is_verified_purchase": [True, True, False, True, True, True],
        }
    )

    ranking = build_product_ranking(reviews)

    assert ranking["confidence_adjusted_score"].between(1, 5).all()


def test_ranking_is_sorted_by_adjusted_score():
    reviews = pd.DataFrame(
        {
            "parent_asin": ["A", "A", "A", "B", "B", "B"],
            "rating": [5, 5, 5, 2, 3, 3],
            "helpful_vote": [0, 0, 0, 0, 0, 0],
            "is_verified_purchase": [True, True, True, True, True, True],
        }
    )

    ranking = build_product_ranking(reviews)

    scores = ranking["confidence_adjusted_score"].tolist()
    assert scores == sorted(scores, reverse=True)