
"""Predict review ratings from review text using TF-IDF and Logistic Regression."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "reviews_clean.csv"
REPORTS_DIR = ROOT / "reports"
RANDOM_STATE = 42


def evaluate(y_true, y_pred):
    """Return evaluation metrics for a set of predictions."""
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(y_true, y_pred)), 4
        ),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "weighted_f1": round(
            float(f1_score(y_true, y_pred, average="weighted")), 4
        ),
    }


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["rating", "text", "parent_asin"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df["rating"] = df["rating"].astype(int)

    # Keep all reviews for a product in the same split to reduce product leakage.
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(
        splitter.split(df, y=df["rating"], groups=df["parent_asin"])
    )

    train = df.iloc[train_idx]
    test = df.iloc[test_idx]

    X_train = train["text"]
    y_train = train["rating"]
    X_test = test["text"]
    y_test = test["rating"]

    # Baseline: always predict the most frequent rating in training data.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train.to_frame(), y_train)
    baseline_pred = baseline.predict(X_test.to_frame())

    # TF-IDF vocabulary is learned from training data only.
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=3,
                    max_features=50000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    baseline_metrics = evaluate(y_test, baseline_pred)
    model_metrics = evaluate(y_test, predictions)
    report = classification_report(
        y_test, predictions, output_dict=True, zero_division=0
    )

    results = {
        "task": "Predict review rating from review text",
        "model": "TF-IDF (unigrams and bigrams) + Logistic Regression",
        "random_state": RANDOM_STATE,
        "split_strategy": (
            "80/20 GroupShuffleSplit grouped by parent_asin; "
            "products do not overlap between train and test"
        ),
        "n_rows": int(len(df)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_train_products": int(train["parent_asin"].nunique()),
        "n_test_products": int(test["parent_asin"].nunique()),
        "product_overlap": int(
            len(set(train["parent_asin"]) & set(test["parent_asin"]))
        ),
        "baseline": baseline_metrics,
        "model_metrics": model_metrics,
        "classification_report": report,
    }

    output_json = REPORTS_DIR / "model_evaluation.json"
    output_json.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    labels = sorted(y_test.unique())
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, label="Number of reviews")
    ax.set(
        title="Review Rating Prediction — Test Set",
        xlabel="Predicted rating",
        ylabel="Actual rating",
        xticks=range(len(labels)),
        yticks=range(len(labels)),
    )

    threshold = matrix.max() / 2 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:,}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(
        REPORTS_DIR / "rating_confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\n--- BASELINE ---")
    print(json.dumps(baseline_metrics, indent=2))
    print("\n--- TF-IDF + LOGISTIC REGRESSION ---")
    print(json.dumps(model_metrics, indent=2))
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"\nMetrics saved to: {output_json}")
    print(f"Confusion matrix saved to: {REPORTS_DIR / 'rating_confusion_matrix.png'}")


if __name__ == "__main__":
    main()