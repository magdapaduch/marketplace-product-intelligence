
"""Rank products using a confidence-adjusted rating score."""

from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/processed/reviews_clean.csv")
OUTPUT_FILE = Path("data/processed/product_ranking.csv")

# A product needs at least this many reviews to enter the shortlist.
MIN_REVIEWS = 3

# Controls how strongly low-volume products are pulled toward
# the overall average rating.
PRIOR_STRENGTH = 5


def build_product_ranking(reviews: pd.DataFrame) -> pd.DataFrame:
    """Create a confidence-adjusted product ranking."""
    required_columns = {
        "parent_asin",
        "rating",
        "helpful_vote",
        "is_verified_purchase",
    }
    missing = required_columns - set(reviews.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    global_mean = reviews["rating"].mean()

    products = (
        reviews.groupby("parent_asin", as_index=False)
        .agg(
            review_count=("rating", "size"),
            average_rating=("rating", "mean"),
            total_helpful_votes=("helpful_vote", "sum"),
            verified_purchase_rate=("is_verified_purchase", "mean"),
        )
    )

    products["confidence_adjusted_score"] = (
        products["review_count"]
        / (products["review_count"] + PRIOR_STRENGTH)
        * products["average_rating"]
        + PRIOR_STRENGTH
        / (products["review_count"] + PRIOR_STRENGTH)
        * global_mean
    )

    # Avoid ranking products on extremely limited evidence.
    products = products.loc[
        products["review_count"] >= MIN_REVIEWS
    ].copy()

    products = products.sort_values(
        ["confidence_adjusted_score", "review_count"],
        ascending=[False, False],
    ).reset_index(drop=True)

    products.insert(0, "rank", range(1, len(products) + 1))

    return products


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_FILE}. Run prepare_data.py first."
        )

    reviews = pd.read_csv(INPUT_FILE)
    ranking = build_product_ranking(reviews)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(OUTPUT_FILE, index=False)

    print(f"Products ranked: {len(ranking):,}")
    print(f"Overall mean rating: {reviews['rating'].mean():.3f}")
    print(f"Saved ranking to: {OUTPUT_FILE}")
    print("\nTop 10 products:")
    print(ranking.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
