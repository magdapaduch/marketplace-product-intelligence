"""Enrich the product ranking with available product metadata."""

import json
from pathlib import Path

import pandas as pd


RANKING_FILE = Path("data/processed/product_ranking.csv")
METADATA_FILE = Path("data/raw/products_sample.jsonl")
OUTPUT_FILE = Path("data/processed/product_ranking_enriched.csv")
REPORT_FILE = Path("data/processed/metadata_match_report.json")


def load_metadata(path: Path) -> pd.DataFrame:
    """Load product metadata from a JSON Lines file."""

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            try:
                product = json.loads(line)
            except json.JSONDecodeError:
                continue

            records.append(
                {
                    "parent_asin": str(product.get("parent_asin", "")).strip(),
                    "product_title": product.get("title"),
                    "main_category": product.get("main_category"),
                    "metadata_average_rating": product.get("average_rating"),
                    "price": product.get("price"),
                }
            )

    metadata = pd.DataFrame(records)

    if metadata.empty:
        return pd.DataFrame(
            columns=[
                "parent_asin",
                "product_title",
                "main_category",
                "metadata_average_rating",
                "price",
            ]
        )

    metadata = metadata.loc[metadata["parent_asin"] != ""]
    metadata = metadata.drop_duplicates(subset=["parent_asin"])

    return metadata


def main() -> None:
    if not RANKING_FILE.exists():
        raise FileNotFoundError(
            f"Missing {RANKING_FILE}. Run rank_products.py first."
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing {METADATA_FILE}. Run download_metadata_sample.py first."
        )

    ranking = pd.read_csv(RANKING_FILE)
    metadata = load_metadata(METADATA_FILE)

    enriched = ranking.merge(
        metadata,
        on="parent_asin",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    matched = int(enriched["_merge"].eq("both").sum())
    total = len(enriched)

    enriched["metadata_matched"] = enriched["_merge"].eq("both")
    enriched = enriched.drop(columns=["_merge"])

    # Keep all ranked products, including those without metadata.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(OUTPUT_FILE, index=False)

    report = {
        "ranked_products": total,
        "metadata_records": len(metadata),
        "matched_products": matched,
        "unmatched_products": total - matched,
        "match_rate": round(matched / total, 4) if total else 0,
        "note": (
            "Metadata was sampled only from the beginning of the source file. "
            "Unmatched products may exist in the full metadata dataset."
        ),
    }

    REPORT_FILE.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Ranked products: {total:,}")
    print(f"Metadata records available: {len(metadata):,}")
    print(f"Products matched: {matched:,}")
    print(f"Match rate: {report['match_rate']:.1%}")
    print(f"Saved enriched ranking to: {OUTPUT_FILE}")
    print(f"Saved match report to: {REPORT_FILE}")

    print("\nExample enriched products:")
    columns = [
        "rank",
        "parent_asin",
        "product_title",
        "main_category",
        "average_rating",
        "confidence_adjusted_score",
        "metadata_matched",
    ]
    print(enriched[columns].head(10).to_string(index=False))


if __name__ == "__main__":
    main()