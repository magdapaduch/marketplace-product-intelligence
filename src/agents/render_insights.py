"""Generate evidence-based business insights from project metrics."""

import json
from pathlib import Path


CONTEXT_FILE = Path("data/processed/insights_context.json")
OUTPUT_FILE = Path("reports/business_insights.md")


def build_insights(context: dict) -> list[dict]:
    """Create business insights and recommendations from measured metrics."""
    insights = []

    ranking = context["ranking"]
    ranked_count = ranking["ranked_products"]
    mean_rating = ranking["mean_sample_rating"]
    top_products = ranking["top_products"]

    insights.append(
        {
            "title": "Ranking coverage",
            "finding": (
                f"The ranking includes {ranked_count:,} products "
                "that meet the minimum review-count threshold."
            ),
            "recommendation": (
                "Use the confidence-adjusted score to prioritize "
                "products for further review, while retaining review "
                "count as a confidence signal."
            ),
        }
    )

    if mean_rating is not None:
        insights.append(
            {
                "title": "Observed product ratings",
                "finding": (
                    f"The mean average rating across ranked products "
                    f"in this sample is {mean_rating:.2f}/5."
                ),
                "recommendation": (
                    "Compare ratings alongside review volume. "
                    "Do not interpret this sample statistic as the "
                    "average rating of the entire marketplace."
                ),
            }
        )

    if top_products:
        top = top_products[0]
        insights.append(
            {
                "title": "Highest-ranked sampled product",
                "finding": (
                    f"Product {top['parent_asin']} ranks first in the "
                    f"exported ranking, with {top['review_count']} "
                    f"reviews, an average rating of "
                    f"{top['average_rating']:.2f}/5, and a "
                    f"confidence-adjusted score of "
                    f"{top['confidence_adjusted_score']:.2f}."
                ),
                "recommendation": (
                    "Review the product and its available metadata "
                    "before using this result in a business decision."
                ),
            }
        )

    trend_count = context["trends"]["flagged_product_months"]
    insights.append(
        {
            "title": "Potential review-count increases",
            "finding": (
                f"The trend detector flagged {trend_count:,} "
                "product-month records for increases in review counts."
            ),
            "recommendation": (
                "Inspect the flagged months and validate the pattern "
                "against a more representative dataset. Review counts "
                "are not direct sales measurements."
            ),
        }
    )

    mapping = context["mapping_quality"]
    issue_count = mapping["records_with_detected_issues"]
    issue_types = mapping.get("issue_counts", {})

    if issue_types:
        issue_summary = ", ".join(
            f"{name}: {count}" for name, count in issue_types.items()
        )
    else:
        issue_summary = "No issues were flagged by the current rules."

    insights.append(
        {
            "title": "Product metadata quality",
            "finding": (
                f"The detector produced {issue_count:,} issue records "
                f"in the sampled metadata. Breakdown: {issue_summary}."
            ),
            "recommendation": (
                "Prioritize missing required metadata and manually "
                "review similarity-based alerts. These signals are "
                "not confirmed mapping errors."
            ),
        }
    )

    return insights


def main():
    if not CONTEXT_FILE.exists():
        raise FileNotFoundError(
            f"Run generate_insights.py first: {CONTEXT_FILE}"
        )

    with CONTEXT_FILE.open(encoding="utf-8") as file:
        context = json.load(file)

    insights = build_insights(context)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Marketplace Product Intelligence: Business Insights",
        "",
        "> Exploratory findings from a byte-position sample of "
        "Amazon Home & Kitchen reviews and sampled product metadata. "
        "Results are not representative of the full marketplace.",
        "",
    ]

    for number, insight in enumerate(insights, start=1):
        lines.extend(
            [
                f"## {number}. {insight['title']}",
                "",
                f"**Finding:** {insight['finding']}",
                "",
                f"**Recommended action:** {insight['recommendation']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Methodological limitations",
            "",
            "- The review data was sampled by byte position, not by "
            "a representative random-sampling design.",
            "- Review counts do not directly measure sales or demand.",
            "- Mapping alerts are candidate data-quality issues, "
            "not verified mapping errors.",
            "- These insights are generated by deterministic rules; "
            "no LLM or external AI API is used.",
            "",
        ]
    )

    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")

    print(f"Generated {len(insights)} insights.")
    print(f"Report saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()