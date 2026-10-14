
"""Test how the product ranking changes with prior strength m."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "processed" / "reviews_clean.csv"
REPORTS_DIR = ROOT / "reports"
PRIOR_STRENGTHS = [3, 5, 10]
MIN_REVIEWS = 3
TOP_N = 10


def rank_products(reviews: pd.DataFrame, prior_strength: int) -> pd.DataFrame:
    """Build a confidence-adjusted ranking for a given prior strength."""
    if prior_strength <= 0:
        raise ValueError("prior_strength must be positive")
    
    global_mean = reviews["rating"].mean()

    products = (
        reviews.groupby("parent_asin", as_index=False)
        .agg(
            review_count=("rating", "size"),
            average_rating=("rating", "mean"),
        )
    )

    m = prior_strength
    products["adjusted_score"] = (
        products["review_count"]
        / (products["review_count"] + m)
        * products["average_rating"]
        + m / (products["review_count"] + m) * global_mean
    )

    products = products.loc[products["review_count"] >= MIN_REVIEWS].copy()
    products = products.sort_values(
        ["adjusted_score", "review_count"],
        ascending=[False, False],
    ).reset_index(drop=True)

    products.insert(0, "rank", range(1, len(products) + 1))
    return products


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    reviews = pd.read_csv(INPUT_FILE)

    rankings = {
        m: rank_products(reviews, m)
        for m in PRIOR_STRENGTHS
    }

    reference_m = 5
    reference_top = set(
        rankings[reference_m].head(TOP_N)["parent_asin"]
    )

    results = {
        "purpose": "Assess sensitivity of product ranking to prior strength",
        "min_reviews": MIN_REVIEWS,
        "top_n": TOP_N,
        "reference_prior_strength": reference_m,
        "rankings": {},
    }

    for m, ranking in rankings.items():
        top_products = ranking.head(TOP_N)
        top_set = set(top_products["parent_asin"])
        overlap = len(top_set & reference_top)

        results["rankings"][str(m)] = {
            "prior_strength": m,
            "ranked_products": int(len(ranking)),
            "top_n_overlap_with_m5": overlap,
            "top_n_overlap_rate_with_m5": round(overlap / TOP_N, 3),
            "top_products": [
                {
                    "rank": int(row.rank),
                    "parent_asin": row.parent_asin,
                    "review_count": int(row.review_count),
                    "average_rating": round(float(row.average_rating), 4),
                    "adjusted_score": round(float(row.adjusted_score), 4),
                }
                for row in top_products.itertuples(index=False)
            ],
        }

    # Compare full rankings for products present in both lists.
    comparisons = {}
    for m in PRIOR_STRENGTHS:
        if m == reference_m:
            continue

        left = rankings[reference_m][["parent_asin", "rank"]].rename(
            columns={"rank": "rank_m5"}
        )
        right = rankings[m][["parent_asin", "rank"]].rename(
            columns={"rank": f"rank_m{m}"}
        )
        common = left.merge(right, on="parent_asin", how="inner")
        corr = spearmanr(common["rank_m5"], common[f"rank_m{m}"]).statistic

        comparisons[f"m5_vs_m{m}"] = {
            "common_products": int(len(common)),
            "spearman_rank_correlation": round(float(corr), 4),
        }

    results["rank_correlation_vs_m5"] = comparisons

    output_json = REPORTS_DIR / "ranking_sensitivity.json"
    output_json.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Visualize top-10 overlap against the reference ranking (m=5).
    ms = PRIOR_STRENGTHS
    overlaps = [
        results["rankings"][str(m)]["top_n_overlap_with_m5"]
        for m in ms
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar([str(m) for m in ms], overlaps)
    ax.set(
        title="Top 10 Ranking Stability Across Prior Strengths",
        xlabel="Prior strength (m)",
        ylabel=f"Products shared with top {TOP_N} at m=5",
        ylim=(0, TOP_N + 1),
    )
    ax.bar_label(bars, fmt="%d")
    fig.tight_layout()
    fig.savefig(
        REPORTS_DIR / "ranking_sensitivity.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\n--- TOP 10 OVERLAP WITH m=5 ---")
    for m in ms:
        info = results["rankings"][str(m)]
        print(
            f"m={m}: {info['top_n_overlap_with_m5']}/{TOP_N} products "
            f"({info['top_n_overlap_rate_with_m5']:.0%})"
        )

    print("\n--- SPEARMAN RANK CORRELATION VS m=5 ---")
    for comparison, info in comparisons.items():
        print(
            f"{comparison}: {info['spearman_rank_correlation']:.4f} "
            f"({info['common_products']:,} common products)"
        )

    print(f"\nDetailed report saved to: {output_json}")
    print(f"Chart saved to: {REPORTS_DIR / 'ranking_sensitivity.png'}")


if __name__ == "__main__":
    main()