"""Prepare evidence-based metrics for business insight generation."""

import json
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/processed")
OUTPUT_FILE = DATA_DIR / "insights_context.json"


def generate_insights_context() -> dict:
    ranking_file = DATA_DIR / "product_ranking.csv"
    trends_file = DATA_DIR / "monthly_trends.csv"
    mapping_file = DATA_DIR / "mapping_issues.csv"
    quality_file = DATA_DIR / "data_quality_report.json"

    for path in [ranking_file, trends_file, mapping_file, quality_file]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required input file not found: {path}"
            )

    ranking = pd.read_csv(ranking_file)
    trends = pd.read_csv(trends_file)
    mapping_issues = pd.read_csv(mapping_file)

    with quality_file.open(encoding="utf-8") as file:
        quality_report = json.load(file)

    context = {
        "project": "Marketplace Product Intelligence",
        "data_scope": (
            "Exploratory analysis of a byte-position sample of "
            "Amazon Home & Kitchen reviews. The sample is not "
            "representative of the full marketplace."
        ),
        "ranking": {
            "ranked_products": int(len(ranking)),
            "minimum_reviews_per_ranked_product": (
                int(ranking["review_count"].min())
                if not ranking.empty else None
            ),
            "mean_sample_rating": (
                round(float(ranking["average_rating"].mean()), 3)
                if not ranking.empty else None
            ),
            "top_products": (
                ranking.head(5)[
                    [
                        "parent_asin",
                        "review_count",
                        "average_rating",
                        "confidence_adjusted_score",
                    ]
                ].to_dict(orient="records")
                if not ranking.empty else []
            ),
        },
        "trends": {
            "flagged_product_months": int(len(trends)),
            "definition": (
                "A flagged product-month has at least 5 reviews in "
                "the current and previous calendar months, at least "
                "10 reviews in the sample overall, and an increase "
                "in current-month review count."
            ),
            "interpretation_warning": (
                "Review-count increases are not direct evidence of "
                "sales or demand growth. Sampling bias may affect "
                "the observed pattern."
            ),
        },
        "mapping_quality": {
            "records_with_detected_issues": int(len(mapping_issues)),
            "issue_counts": (
                {
                    str(key): int(value)
                    for key, value in mapping_issues[
                        "issue_reason"
                    ].str.split(":").str[0].value_counts().items()
                }
                if not mapping_issues.empty else {}
            ),
            "interpretation_warning": (
                "These are potential data-quality signals, not "
                "confirmed incorrect product mappings."
            ),
        },
        "data_quality": quality_report,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(context, file, indent=2, ensure_ascii=False, default=str)

    return context


if __name__ == "__main__":
    result = generate_insights_context()
    print(f"Saved metrics context to: {OUTPUT_FILE}")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))