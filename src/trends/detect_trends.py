"""Detect exploratory monthly increases in product review activity."""

from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/processed/reviews_clean.csv")
OUTPUT_FILE = Path("data/processed/monthly_trends.csv")

MIN_MONTHLY_REVIEWS = 5
MIN_TOTAL_REVIEWS = 10


def detect_trends(reviews: pd.DataFrame) -> pd.DataFrame:
    """Compare review activity across consecutive calendar months."""

    required_columns = {"parent_asin", "timestamp", "rating"}
    missing = required_columns - set(reviews.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data = reviews.copy()
    data["timestamp"] = pd.to_datetime(
        data["timestamp"], utc=True, errors="coerce"
    )
    data["rating"] = pd.to_numeric(data["rating"], errors="coerce")
    data["parent_asin"] = data["parent_asin"].astype("string").str.strip()

    data = data.dropna(subset=["parent_asin", "timestamp", "rating"])
    data = data.loc[data["parent_asin"] != ""]

    # Use UTC consistently, then remove timezone information to define
    # calendar-month labels without a pandas timezone warning.
    data["review_month"] = (
        data["timestamp"]
        .dt.tz_localize(None)
        .dt.to_period("M")
    )

    monthly = (
        data.groupby(["parent_asin", "review_month"], as_index=False)
        .agg(
            monthly_review_count=("rating", "size"),
            monthly_average_rating=("rating", "mean"),
        )
    )

    # Build a continuous calendar-month timeline for each product.
    # Missing months are filled with zero observed reviews in this sample.
    all_months = pd.period_range(
        data["review_month"].min(),
        data["review_month"].max(),
        freq="M",
    )
    product_ids = monthly["parent_asin"].unique()

    full_index = pd.MultiIndex.from_product(
        [product_ids, all_months],
        names=["parent_asin", "review_month"],
    )

    monthly = (
        monthly.set_index(["parent_asin", "review_month"])
        .reindex(full_index)
        .reset_index()
    )

    monthly["monthly_review_count"] = (
        monthly["monthly_review_count"].fillna(0).astype(int)
    )

    monthly = monthly.sort_values(
        ["parent_asin", "review_month"]
    ).reset_index(drop=True)

    monthly["previous_month_reviews"] = (
        monthly.groupby("parent_asin")["monthly_review_count"].shift(1)
    )

    monthly["review_count_change"] = (
        monthly["monthly_review_count"] - monthly["previous_month_reviews"]
    )

    monthly["review_count_growth_rate"] = (
        monthly["review_count_change"]
        / monthly["previous_month_reviews"].where(
            monthly["previous_month_reviews"] > 0
        )
    )

    total_reviews = monthly.groupby("parent_asin")[
        "monthly_review_count"
    ].transform("sum")
    monthly["total_sample_reviews"] = total_reviews

    monthly["potential_increase"] = (
        (monthly["monthly_review_count"] >= MIN_MONTHLY_REVIEWS)
        & (monthly["total_sample_reviews"] >= MIN_TOTAL_REVIEWS)
        & (monthly["previous_month_reviews"] >= MIN_MONTHLY_REVIEWS)
        & (monthly["review_count_change"] > 0)
    )

    # Restore readable YYYY-MM strings for CSV output.
    monthly["review_month"] = monthly["review_month"].astype(str)

    monthly = monthly.sort_values(
        ["potential_increase", "review_count_change"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return monthly


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_FILE}. Run prepare_data.py first."
        )

    reviews = pd.read_csv(INPUT_FILE)
    trends = detect_trends(reviews)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    trends.to_csv(OUTPUT_FILE, index=False)

    candidates = trends.loc[trends["potential_increase"]].head(20)

    print(f"Product-month records analyzed: {len(trends):,}")
    print(
        "Potential monthly increases flagged: "
        f"{trends['potential_increase'].sum():,}"
    )
    print(f"Saved results to: {OUTPUT_FILE}")
    print("\nTop exploratory candidates:")

    if candidates.empty:
        print("No candidates met the current thresholds.")
    else:
        print(
            candidates[
                [
                    "parent_asin",
                    "review_month",
                    "previous_month_reviews",
                    "monthly_review_count",
                    "review_count_change",
                    "review_count_growth_rate",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()