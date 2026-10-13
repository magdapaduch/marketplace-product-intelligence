# Marketplace Product Intelligence

An exploratory data science project for product ranking, review-activity trend detection, product metadata quality checks, and automated business insights.

## Project Overview

Online marketplaces need reliable ways to prioritize products, monitor changes in customer feedback, and identify potential product data-quality issues.

This project builds a reproducible analytical pipeline using Amazon Home & Kitchen review data. It combines statistical ranking, monthly activity analysis, explainable data-quality rules, and automated reporting.

**The project is an exploratory prototype, not a production marketplace monitoring system.**

## Business Questions

- How can products be ranked while accounting for differences in review volume?
- Which products show increases in observed monthly review activity?
- Which sampled product records have potential metadata quality issues?
- How can analytical results be converted into actionable business recommendations?
- What limitations must be considered before using these results in business decisions?

## Dataset

**Source:** [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/main.html), Home & Kitchen category.

The review dataset is very large, so this project uses bounded HTTP byte-range sampling instead of downloading the full source file. Product metadata is sampled separately.

Relevant fields include:

- `parent_asin` — product identifier used for aggregation and metadata matching.
- `rating` — review rating.
- `timestamp` — review timestamp.
- `helpful_vote` — helpfulness votes.
- `verified_purchase` — verified-purchase indicator.
- Product metadata such as `title`, `main_category`, `features`, and `description`.

Raw data and processed data files are excluded from version control.

## Analytical Pipeline

1. **Data sampling** — retrieve bounded sections of the source review file.
2. **Data preparation** — validate fields, normalize types and timestamps, remove duplicates, and create monthly features.
3. **Product ranking** — calculate product-level statistics and a confidence-adjusted rating score.
4. **Trend detection** — flag increases in observed monthly review counts.
5. **Metadata enrichment** — match sampled metadata to products in the ranking.
6. **Mapping-quality signals** — flag missing metadata and highly similar product titles appearing in different categories.
7. **Business insight reporting** — convert measured results into findings and recommendations.
8. **Testing** — validate key ranking, detection, and insight-generation behavior.

## Product Ranking

The ranking uses a Bayesian-style weighted rating:

\[
\text{score} =
\frac{v}{v+m}R + \frac{m}{v+m}C
\]

Where:

- \(v\) is the product's review count in the cleaned sample.
- \(R\) is the product's average rating in the sample.
- \(m\) is the prior strength, set to 5.
- \(C\) is the overall mean rating in the cleaned sample.

Products must have at least three sampled reviews to be included.

The weighted score balances each product's observed average rating with the overall sample mean. This reduces the influence of products with very few reviews. It does not guarantee that a product is objectively better.

This is a statistical ranking heuristic, **not a trained machine learning model**.

Implementation: `src/models/rank_products.py`

## Review-Activity Trend Detection

The trend pipeline aggregates reviews by product and calendar month. It fills missing product-month combinations with zero observed reviews, enabling comparisons between consecutive calendar months.

A potential increase is flagged when:

- The current month has at least 5 sampled reviews.
- The previous calendar month has at least 5 sampled reviews.
- The product has at least 10 sampled reviews across the analyzed period.
- The current month's review count exceeds the previous month's count.

Implementation: `src/trends/detect_trends.py`

These flags identify candidates for further investigation. Review activity is not a direct measurement of sales, demand, conversion, or revenue.

## Product Metadata Quality Checks

The mapping-quality prototype currently checks for:

- Missing product titles.
- Missing main categories.
- Highly similar product titles assigned to different categories.

Title similarity is measured using character-level TF-IDF features and cosine similarity. The initial similarity threshold is 0.88.

The detector produces a reason for each flagged record to support review and investigation.

Implementation: `src/features/detect_mapping_issues.py`

**Important limitations:**

- These rules identify potential data-quality issues, not confirmed incorrect product mappings.
- The current metadata sample is small and has limited overlap with ranked products.
- Similar titles in different categories may be legitimate, and different titles do not necessarily indicate different products.
- The similarity threshold is a configurable starting point, not a validated decision boundary.
- The current prototype has not been evaluated against a labeled dataset of verified mapping errors.

The current metadata sample produced 3 missing-category issue records among 266 sampled products. No cross-category title-similarity alerts were found in that sample.

## Automated Business Insights

The insight pipeline reads the generated ranking, trend, metadata-quality, and data-quality outputs. It produces findings and recommended actions based on measured metrics.

The generated report includes:

- Ranking coverage and observed rating statistics.
- The highest-ranked product in the sample.
- The number of flagged product-month increases.
- Detected metadata-quality issues.
- Methodological limitations and recommended follow-up actions.

Implementation:

- `src/agents/generate_insights.py` — prepares structured metrics.
- `src/agents/render_insights.py` — generates the Markdown report.

Report output: `reports/business_insights.md`

**The current insight generator is deterministic and rule-based. It does not use an LLM or an external AI API.** An LLM-based extension could be added later, with outputs grounded in validated metrics and explicit uncertainty.

## Current Results

Results from the current local sample:

| Metric | Result |
|---|---:|
| Raw review records sampled | 44,918 |
| Cleaned review records | 44,532 |
| Duplicate records removed | 386 |
| Unique products in cleaned reviews | 33,359 |
| Products included in ranking | 1,348 |
| Potential monthly increases flagged | 37 |
| Metadata records scanned | 26,148 |
| Sampled metadata products saved | 266 |
| Ranked products matched to sampled metadata | 20 of 1,348 (1.5%) |
| Missing-category issue records | 3 |
| Automated tests passing | 8 |

The mean rating across the cleaned review sample is approximately 4.162.

These figures describe the current local sample and may change if the sampling procedure or source data changes.

## Data Limitations

- **Non-representative sampling:** reviews were sampled from byte offsets in a large JSONL file. This is not a representative random sample and may introduce selection, ordering, and time bias.
- **Trend interpretation:** increases refer to observed review activity in the sample, not confirmed marketplace-wide trends.
- **Missing months:** zero observed reviews for a product-month do not prove that no reviews existed in the source dataset.
- **Incomplete metadata:** only 20 ranked products matched the sampled metadata. An unmatched product should not be treated as nonexistent.
- **Mapping-quality evaluation:** the current rules have not been validated against manually verified mapping labels.
- **Insight generation:** the current report is generated using deterministic rules, not a generative language model.
- **Causal inference:** the analysis does not establish why review activity changes.

All findings should be treated as exploratory rather than production-ready business metrics.

## Repository Structure

```text
marketplace-product-intelligence/
├── data/
│   ├── raw/                  # Local samples; excluded from Git
│   └── processed/            # Generated outputs; excluded from Git
├── reports/
│   └── business_insights.md
├── src/
│   ├── agents/
│   │   ├── generate_insights.py
│   │   └── render_insights.py
│   ├── data/
│   │   ├── download_data.py
│   │   ├── download_metadata_sample.py
│   │   ├── enrich_product_ranking.py
│   │   └── prepare_data.py
│   ├── features/
│   │   └── detect_mapping_issues.py
│   ├── models/
│   │   └── rank_products.py
│   └── trends/
│       └── detect_trends.py
├── tests/
│   ├── test_detect_mapping_issues.py
│   ├── test_rank_products.py
│   └── test_render_insights.py
├── .gitignore
├── pyproject.toml
└── README.md
```

## Setup

Requires Python 3.11 or 3.12.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
python -m pip install -e ".[data,dev]"
```

## Run the Pipeline

Run commands from the repository root.

Sample and prepare the review data:

```bash
python src/data/download_data.py
python src/data/prepare_data.py
```

Build the product ranking and detect review-activity increases:

```bash
python src/models/rank_products.py
python src/trends/detect_trends.py
```

Sample and enrich product metadata:

```bash
python src/data/download_metadata_sample.py
python src/data/enrich_product_ranking.py
```

Run the mapping-quality checks:

```bash
python src/features/detect_mapping_issues.py
```

Generate the insight context and report:

```bash
python src/agents/generate_insights.py
python src/agents/render_insights.py
```

Run automated tests:

```bash
python -m pytest -v
```

The pipeline expects the source samples to be available locally. Generated files are written under `data/raw/` and `data/processed/`, which are intentionally excluded from version control.

## Tech Stack

- Python
- pandas and NumPy
- scikit-learn
- pytest
- Git

## Future Improvements

- Evaluate mapping-quality rules against a labeled set of verified examples.
- Improve metadata coverage and investigate identifier consistency.
- Add visualizations for ranking and review-activity changes.
- Expand tests for data preparation and trend detection.
- Add SQL examples for product-level and monthly aggregations.
- Evaluate more robust trend-detection methods.
- Explore an LLM-based insight-generation layer grounded in structured analytical outputs.

## Author's Note

This project demonstrates an end-to-end exploratory workflow: data sampling, data preparation, statistical ranking, trend detection, explainable quality checks, automated reporting, and testing.

Its focus is on reproducibility, transparent assumptions, and careful interpretation of imperfect data.