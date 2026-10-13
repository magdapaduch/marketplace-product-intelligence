"""Sample product metadata from multiple positions in a large JSONL file."""

import json
from pathlib import Path

import pandas as pd
import requests


REVIEWS_FILE = Path("data/processed/reviews_clean.csv")
OUTPUT_FILE = Path("data/raw/products_sample.jsonl")

METADATA_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/"
    "Amazon-Reviews-2023/resolve/main/raw/meta_categories/"
    "meta_Home_and_Kitchen.jsonl"
)

CHUNK_BYTES = 8 * 1024 * 1024
SAMPLE_POSITIONS = [0.00, 0.10, 0.20, 0.30, 0.40,
                    0.50, 0.60, 0.70, 0.80, 0.90]


def main() -> None:
    if not REVIEWS_FILE.exists():
        raise FileNotFoundError(
            f"Missing {REVIEWS_FILE}. Run prepare_data.py first."
        )

    reviews = pd.read_csv(REVIEWS_FILE, usecols=["parent_asin"])
    target_ids = set(reviews["parent_asin"].dropna().astype(str).str.strip())

    # HEAD retrieves file size without downloading the full dataset.
    head = requests.head(METADATA_URL, allow_redirects=True, timeout=60)
    head.raise_for_status()
    file_size = int(head.headers["Content-Length"])

    found = {}
    records_scanned = 0
    bytes_read = 0

    for position in SAMPLE_POSITIONS:
        start = int(file_size * position)
        end = min(start + CHUNK_BYTES - 1, file_size - 1)

        response = requests.get(
            METADATA_URL,
            headers={"Range": f"bytes={start}-{end}"},
            timeout=60,
        )
        response.raise_for_status()

        if response.status_code != 206:
            raise RuntimeError(
                "Server did not honor the byte-range request. "
                "Stopped to avoid downloading the full metadata file."
            )

        content = response.content
        bytes_read += len(content)
        lines = content.split(b"\n")

        # An arbitrary byte offset can start in the middle of a JSON record.
        if start > 0:
            lines = lines[1:]

        # The last line may be incomplete because the byte range ended mid-record.
        lines = lines[:-1]

        for line in lines:
            if not line.strip():
                continue

            try:
                product = json.loads(line)
            except json.JSONDecodeError:
                continue

            records_scanned += 1
            product_id = str(product.get("parent_asin", "")).strip()

            if product_id in target_ids and product_id not in found:
                found[product_id] = product

        print(
            f"Scanned position {position:.0%}: "
            f"{records_scanned:,} records total, "
            f"{len(found):,} matching products"
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as output:
        for product in found.values():
            output.write(json.dumps(product, ensure_ascii=False) + "\n")

    print(f"\nMetadata file size: {file_size:,} bytes")
    print(f"Bytes sampled: {bytes_read:,}")
    print(f"Records scanned: {records_scanned:,}")
    print(f"Unique matching products saved: {len(found):,}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("Note: this is a distributed sample, not a complete metadata catalog.")


if __name__ == "__main__":
    main()