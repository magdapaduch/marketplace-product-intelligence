from src.agents.render_insights import build_insights


def test_build_insights_uses_provided_metrics():
    context = {
        "ranking": {
            "ranked_products": 12,
            "mean_sample_rating": 4.25,
            "top_products": [
                {
                    "parent_asin": "TEST123",
                    "review_count": 20,
                    "average_rating": 4.8,
                    "confidence_adjusted_score": 4.67,
                }
            ],
        },
        "trends": {"flagged_product_months": 3},
        "mapping_quality": {
            "records_with_detected_issues": 2,
            "issue_counts": {"missing_main_category": 2},
        },
    }

    insights = build_insights(context)
    report_text = " ".join(
        item["finding"] + " " + item["recommendation"]
        for item in insights
    )

    assert len(insights) == 5
    assert "12 products" in report_text
    assert "4.25/5" in report_text
    assert "3 product-month" in report_text
    assert "missing_main_category: 2" in report_text


def test_insights_include_recommendations():
    context = {
        "ranking": {
            "ranked_products": 0,
            "mean_sample_rating": None,
            "top_products": [],
        },
        "trends": {"flagged_product_months": 0},
        "mapping_quality": {
            "records_with_detected_issues": 0,
            "issue_counts": {},
        },
    }

    insights = build_insights(context)

    assert insights
    assert all(
        item["recommendation"].strip()
        for item in insights
    )