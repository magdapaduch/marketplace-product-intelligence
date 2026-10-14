"""Create polished portfolio visualizations."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "figures"


def setup_chart():
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        palette="viridis",
        font_scale=1.0,
    )
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 220
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["axes.labelsize"] = 11


def save_chart(fig, filename):
    fig.savefig(OUT / filename, facecolor="white")
    plt.close(fig)


def plot_rating_distribution():
    reviews = pd.read_csv(DATA / "reviews_clean.csv")
    reviews["rating"] = pd.to_numeric(
        reviews["rating"], errors="coerce"
    )
    reviews = reviews.dropna(subset=["rating"])
    reviews["rating"] = reviews["rating"].astype(int)

    order = [1, 2, 3, 4, 5]
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.countplot(
        data=reviews,
        x="rating",
        order=order,
        hue="rating",
        palette="viridis",
        legend=False,
        ax=ax,
    )

    total = len(reviews)
    for bar in ax.patches:
        count = bar.get_height()
        if count > 0:
            ax.annotate(
                f"{count:,.0f}\n({count / total:.1%})",
                (bar.get_x() + bar.get_width() / 2, count),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 4),
                textcoords="offset points",
            )

    ax.set(
        title="Review Rating Distribution",
        xlabel="Rating (stars)",
        ylabel="Number of reviews",
    )
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", alpha=0.25)
    sns.despine(ax=ax)
    fig.tight_layout()
    save_chart(fig, "rating_distribution.png")


def plot_monthly_review_activity():
    monthly = pd.read_csv(DATA / "monthly_overview.csv")
    monthly["review_month"] = pd.to_datetime(
        monthly["review_month"], format="%Y-%m", errors="coerce"
    )
    monthly["review_count"] = pd.to_numeric(
        monthly["review_count"], errors="coerce"
    )
    monthly = monthly.dropna(
        subset=["review_month", "review_count"]
    ).sort_values("review_month")

    fig, ax = plt.subplots(figsize=(11, 5))

    sns.lineplot(
        data=monthly,
        x="review_month",
        y="review_count",
        color=sns.color_palette("viridis", 6)[4],
        linewidth=1.8,
        ax=ax,
    )

    ax.set(
        title="Monthly Review Activity in the Collected Sample",
        xlabel="Month",
        ylabel="Observed reviews",
    )
    ax.text(
        0.01,
        0.97,
        "Sampled records only — not total marketplace demand",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color="dimgray",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    save_chart(fig, "monthly_review_activity.png")


def plot_top_products():
    ranking = pd.read_csv(DATA / "product_ranking.csv")
    ranking["rank"] = pd.to_numeric(
        ranking["rank"], errors="coerce"
    )
    ranking = ranking.dropna(subset=["rank"])
    top = ranking.nsmallest(10, "rank").copy()
    top = top.sort_values("confidence_adjusted_score")

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=top,
        x="confidence_adjusted_score",
        y="parent_asin",
        hue="confidence_adjusted_score",
        palette="viridis",
        legend=False,
        ax=ax,
    )

    for bar in ax.patches:
        width = bar.get_width()
        ax.annotate(
            f"{width:.3f}",
            (width, bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )

    ax.set(
        title="Top 10 Products by Confidence-Adjusted Score",
        xlabel="Adjusted score (1–5)",
        ylabel="Product ID",
    )
    ax.set_xlim(0, 5.3)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    save_chart(fig, "top_products_ranking.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    setup_chart()

    plot_rating_distribution()
    plot_monthly_review_activity()
    plot_top_products()

    print(f"Charts saved to: {OUT}")
    for path in sorted(OUT.glob("*.png")):
        print(f"- {path.name}")


if __name__ == "__main__":
    main()