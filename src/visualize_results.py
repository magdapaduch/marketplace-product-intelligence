from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "figures"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Distribution of review ratings
    reviews = pd.read_csv(DATA / "reviews_clean.csv")
    ratings = reviews["rating"].value_counts().reindex(
        [1, 2, 3, 4, 5], fill_value=0
    ).sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(ratings.index.astype(str), ratings.values)
    ax.bar_label(bars, padding=3)
    ax.set_title("Distribution of Review Ratings")
    ax.set_xlabel("Rating (stars)")
    ax.set_ylabel("Number of reviews")
    fig.tight_layout()
    fig.savefig(OUT / "rating_distribution.png", dpi=180)
    plt.close(fig)

    # 2. Monthly review activity in the collected sample
    monthly = pd.read_csv(DATA / "monthly_overview.csv")
    monthly["review_month"] = pd.to_datetime(
        monthly["review_month"], format="%Y-%m", errors="coerce"
    )
    monthly = monthly.dropna(subset=["review_month"])
    monthly = monthly.sort_values("review_month")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        monthly["review_month"],
        monthly["review_count"],
        marker=".",
        linewidth=1,
    )
    ax.set_title("Monthly Review Counts in the Collected Sample")
    ax.set_xlabel("Month")
    ax.set_ylabel("Observed reviews")
    ax.text(
        0.01, 0.97,
        "Sample only — not total marketplace demand",
        transform=ax.transAxes,
        va="top",
    )
    fig.tight_layout()
    fig.savefig(OUT / "monthly_review_activity.png", dpi=180)
    plt.close(fig)

    # 3. Top 10 products by confidence-adjusted score
    ranking = pd.read_csv(DATA / "product_ranking.csv")
    top = ranking.nsmallest(10, "rank").sort_values(
        "confidence_adjusted_score"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        top["parent_asin"].astype(str),
        top["confidence_adjusted_score"],
    )
    ax.set_xlim(0, 5)
    ax.set_title("Top 10 Products by Confidence-Adjusted Score")
    ax.set_xlabel("Adjusted score (1–5)")
    ax.set_ylabel("Product ID")
    fig.tight_layout()
    fig.savefig(OUT / "top_products_ranking.png", dpi=180)
    plt.close(fig)

    print("Charts saved to:", OUT)
    print("1. rating_distribution.png")
    print("2. monthly_review_activity.png")
    print("3. top_products_ranking.png")


if __name__ == "__main__":
    main()