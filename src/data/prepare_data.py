
"""Validate and transform the exploratory review sample."""

import json
from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/raw/reviews_sample.jsonl")
OUTPUT_DIR = Path("data/processed")

REQUIRED_COLUMNS = {
    "rating",
    "title",
    "text",
    "parent_asin",
    "timestamp",
    "helpful_vote",
    "verified_purchase",
}


def load_reviews(path: Path) -> pd.DataFrame:
    """Load JSONL reviews and validate the expected schema."""
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}. Run download_data.py first."
        )

    df = pd.read_json(path, lines=True)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    return df


def clean_reviews(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean review records and create a data quality summary."""
    df = df.copy()
    initial_rows = len(df)

    # Convert fields to consistent data types.
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["helpful_vote"] = pd.to_numeric(
        df["helpful_vote"], errors="coerce"
    ).fillna(0)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True, errors="coerce"
    )

    df["parent_asin"] = df["parent_asin"].astype("string").str.strip()
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["text"] = df["text"].fillna("").astype(str).str.strip()

    # A valid review needs a product ID, timestamp, and rating in [1, 5].
    valid_mask = (
        df["parent_asin"].notna()
        & df["parent_asin"].ne("")
        & df["timestamp"].notna()
        & df["rating"].between(1, 5)
    )
    invalid_rows = int((~valid_mask).sum())
    df = df.loc[valid_mask].copy()

    # Remove exact duplicates across the fields available in this dataset.
    # This is a practical heuristic, not a guaranteed unique review ID.
    duplicate_subset = [
        "user_id",
        "parent_asin",
        "timestamp",
        "rating",
        "title",
        "text",
    ]
    available_subset = [
        col for col in duplicate_subset if col in df.columns
    ]
    duplicate_rows = int(df.duplicated(subset=available_subset).sum())
    df = df.drop_duplicates(subset=available_subset).copy()

    df["review_month"] = (
        df["timestamp"].dt.tz_convert(None).dt.to_period("M").astype(str)
    )
    df["is_verified_purchase"] = (
        df["verified_purchase"].astype("boolean").fillna(False).astype(bool)
    )

    quality_report = {
        "input_rows": initial_rows,
        "invalid_rows_removed": invalid_rows,
        "duplicate_rows_removed": duplicate_rows,
        "output_rows": len(df),
        "unique_products": int(df["parent_asin"].nunique()),
        "timestamp_min_utc": df["timestamp"].min().isoformat(),
        "timestamp_max_utc": df["timestamp"].max().isoformat(),
        "rating_min": float(df["rating"].min()),
        "rating_max": float(df["rating"].max()),
        "missing_values_after_cleaning": {
            key: int(value)
            for key, value in df.isna().sum().items()
        },
    }

    return df, quality_report


def build_aggregations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate reviews by product-month and month."""
    product_monthly = (
        df.groupby(["parent_asin", "review_month"], as_index=False)
        .agg(
            review_count=("rating", "size"),
            average_rating=("rating", "mean"),
            rating_std=("rating", "std"),
            helpful_vote_sum=("helpful_vote", "sum"),
            verified_purchase_rate=("is_verified_purchase", "mean"),
        )
        .sort_values(["review_month", "review_count"], ascending=[True, False])
    )

    monthly_overview = (
        df.groupby("review_month", as_index=False)
        .agg(
            review_count=("rating", "size"),
            unique_products=("parent_asin", "nunique"),
            average_rating=("rating", "mean"),
            verified_purchase_rate=("is_verified_purchase", "mean"),
        )
        .sort_values("review_month")
    )

    return product_monthly, monthly_overview


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reviews = load_reviews(INPUT_FILE)
    clean_df, quality_report = clean_reviews(reviews)

    product_monthly, monthly_overview = build_aggregations(clean_df)

    clean_df.to_csv(OUTPUT_DIR / "reviews_clean.csv", index=False)
    product_monthly.to_csv(
        OUTPUT_DIR / "product_monthly.csv", index=False
    )
    monthly_overview.to_csv(
        OUTPUT_DIR / "monthly_overview.csv", index=False
    )

    with (OUTPUT_DIR / "data_quality_report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(quality_report, file, indent=2, ensure_ascii=False)

    print("Pipeline completed successfully.")
    print(f"Clean reviews: {len(clean_df):,}")
    print(f"Unique products: {clean_df['parent_asin'].nunique():,}")
    print(f"Product-month rows: {len(product_monthly):,}")
    print(f"Months covered: {clean_df['review_month'].nunique():,}")
    print(
        f"Date range: {quality_report['timestamp_min_utc']} "
        f"to {quality_report['timestamp_max_utc']}"
    )
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
