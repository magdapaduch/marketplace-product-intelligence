"""Detect potential product-mapping issues in sampled product metadata."""

from pathlib import Path
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


INPUT_FILE = Path("data/raw/products_sample.jsonl")
OUTPUT_FILE = Path("data/processed/mapping_issues.csv")
SIMILARITY_THRESHOLD = 0.88


def normalize_title(title: object) -> str:
    """Normalize a title for comparison."""
    if pd.isna(title):
        return ""
    return re.sub(r"\s+", " ", str(title).lower()).strip()


def detect_mapping_issues(products: pd.DataFrame) -> pd.DataFrame:
    """Flag missing metadata and similar titles in different categories."""
    required = {"parent_asin", "title", "main_category"}
    missing = required - set(products.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = products.copy()
    df["normalized_title"] = df["title"].map(normalize_title)

    issues = []

    def add_issue(index, reason, similarity=None):
        row = df.loc[index]
        issues.append(
            {
                "parent_asin": row["parent_asin"],
                "title": row["title"],
                "main_category": row["main_category"],
                "issue_reason": reason,
                "similarity_score": similarity,
            }
        )

    # Rule 1: Missing key metadata.
    for index, row in df.iterrows():
        if not row["normalized_title"]:
            add_issue(index, "missing_product_title")

        if pd.isna(row["main_category"]) or not str(
            row["main_category"]
        ).strip():
            add_issue(index, "missing_main_category")

    # Rule 2: Similar product titles assigned to different categories.
    valid = df[
        (df["normalized_title"].str.len() >= 10)
        & df["main_category"].notna()
    ].copy()

    valid["main_category"] = valid["main_category"].astype(str)

    if len(valid) >= 2:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
        )
        matrix = vectorizer.fit_transform(valid["normalized_title"])
        similarities = cosine_similarity(matrix)

        indices = list(valid.index)

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                left_index = indices[i]
                right_index = indices[j]

                left = df.loc[left_index]
                right = df.loc[right_index]
                score = similarities[i, j]

                if (
                    score >= SIMILARITY_THRESHOLD
                    and left["main_category"] != right["main_category"]
                ):
                    add_issue(
                        left_index,
                        f"similar_title_in_different_category:"
                        f"{right['parent_asin']}",
                        round(float(score), 3),
                    )
                    add_issue(
                        right_index,
                        f"similar_title_in_different_category:"
                        f"{left['parent_asin']}",
                        round(float(score), 3),
                    )

    columns = [
        "parent_asin",
        "title",
        "main_category",
        "issue_reason",
        "similarity_score",
    ]
    result = pd.DataFrame(issues, columns=columns)

    if not result.empty:
        result = result.drop_duplicates().reset_index(drop=True)

    return result


def main():
    products = pd.read_json(INPUT_FILE, lines=True)
    result = detect_mapping_issues(products)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print(f"Products analyzed: {len(products)}")
    print(f"Potential issues found: {len(result)}")
    print(f"Saved to: {OUTPUT_FILE}")

    if not result.empty:
        print("\nIssue types:")
        print(result["issue_reason"].str.split(":").str[0].value_counts())


if __name__ == "__main__":
    main()