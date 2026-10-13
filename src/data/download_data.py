
"""Download a bounded exploratory sample of Amazon Reviews 2023."""

import json
from pathlib import Path

import requests

DATA_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/"
    "Amazon-Reviews-2023/resolve/main/raw/review_categories/"
    "Home_and_Kitchen.jsonl"
)

OUTPUT_FILE = Path("data/raw/reviews_sample.jsonl")
CHUNK_SIZE = 2 * 1024 * 1024  # 2 MiB per request
SAMPLE_POSITIONS = [0.05, 0.15, 0.25, 0.35, 0.45,
                    0.55, 0.65, 0.75, 0.85, 0.95]


def get_file_size() -> int:
    """Read the remote file size using a one-byte range request."""
    response = requests.get(
        DATA_URL,
        headers={"Range": "bytes=0-0"},
        timeout=30,
    )
    response.raise_for_status()

    content_range = response.headers.get("Content-Range", "")
    if response.status_code != 206 or "/" not in content_range:
        raise RuntimeError(
            "The server did not confirm byte-range support. "
            f"Status={response.status_code}, Content-Range={content_range!r}"
        )

    return int(content_range.rsplit("/", 1)[1])


def fetch_chunk(start: int, end: int) -> bytes:
    """Fetch one bounded byte range and verify the server response."""
    response = requests.get(
        DATA_URL,
        headers={"Range": f"bytes={start}-{end}"},
        timeout=60,
    )
    response.raise_for_status()

    if response.status_code != 206:
        raise RuntimeError(
            f"Expected HTTP 206 for byte range {start}-{end}, "
            f"received {response.status_code}"
        )

    return response.content


def extract_records(
    content: bytes,
    discard_first_line: bool,
    is_final_chunk: bool,
) -> list[dict]:
    """Parse complete JSON Lines records from a byte chunk."""
    lines = content.splitlines()

    if discard_first_line and lines:
        # The first line may begin halfway through a JSON record.
        lines = lines[1:]

    if not is_final_chunk and content and not content.endswith(b"\n"):
        # The final line may be truncated at the end of the byte range.
        lines = lines[:-1]

    records = []
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        if isinstance(record, dict):
            records.append(record)

    return records


def main() -> None:
    file_size = get_file_size()
    print(f"Remote file size: {file_size:,} bytes")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    total_records = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as output:
        for position in SAMPLE_POSITIONS:
            start = int((file_size - CHUNK_SIZE) * position)
            end = min(start + CHUNK_SIZE - 1, file_size - 1)

            print(
                f"Fetching {position:.0%} of file "
                f"(bytes {start:,}-{end:,})..."
            )

            content = fetch_chunk(start, end)
            records = extract_records(
                content,
                discard_first_line=(start > 0),
                is_final_chunk=(end == file_size - 1),
            )

            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_records += len(records)
            print(f"  Valid records: {len(records):,}")

    print(f"\nSaved {total_records:,} records to {OUTPUT_FILE}")
    print("Note: this is an exploratory sample, not a random sample.")


if __name__ == "__main__":
    main()