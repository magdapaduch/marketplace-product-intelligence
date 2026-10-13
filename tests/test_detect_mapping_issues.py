import pandas as pd

from src.features.detect_mapping_issues import detect_mapping_issues


def test_detects_missing_category():
    products = pd.DataFrame(
        [
            {
                "parent_asin": "A1",
                "title": "Stainless Steel Water Bottle",
                "main_category": None,
            }
        ]
    )

    result = detect_mapping_issues(products)

    assert "missing_main_category" in result["issue_reason"].values


def test_clean_product_has_no_issues():
    products = pd.DataFrame(
        [
            {
                "parent_asin": "A1",
                "title": "Stainless Steel Water Bottle",
                "main_category": "Kitchen",
            }
        ]
    )

    result = detect_mapping_issues(products)

    assert result.empty


def test_detects_similar_titles_in_different_categories():
    products = pd.DataFrame(
        [
            {
                "parent_asin": "A1",
                "title": "Stainless Steel Insulated Water Bottle 750ml",
                "main_category": "Kitchen",
            },
            {
                "parent_asin": "A2",
                "title": "Stainless Steel Insulated Water Bottle 750ml",
                "main_category": "Pet Supplies",
            },
        ]
    )

    result = detect_mapping_issues(products)

    assert result["issue_reason"].str.startswith(
        "similar_title_in_different_category"
    ).any()