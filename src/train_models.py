from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.data_loader import load_kaggle_fake_news_dataset


@dataclass
class ModelResult:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float


def build_vectorizer(
    feature_type: str,
    ngram_max: int,
    min_df: int,
    max_features: int | None,
):
    """Returns a vectorizer based on feature_type.

    feature_type:
      - 'tfidf' : TF-IDF features
      - 'bow'   : Bag-of-Words (CountVectorizer)
    """
    feature_type = feature_type.lower().strip()
    if feature_type == "tfidf":
        return TfidfVectorizer(
            stop_words="english",
            max_df=0.9,
            min_df=min_df,
            ngram_range=(1, ngram_max),
            max_features=max_features,
            sublinear_tf=True,
        )
    if feature_type == "bow":
        return CountVectorizer(
            stop_words="english",
            max_df=0.9,
            min_df=min_df,
            ngram_range=(1, ngram_max),
            max_features=max_features,
        )
    raise ValueError("feature_type must be one of: tfidf, bow")


def build_nb_pipeline(feature_type: str, ngram_max: int, min_df: int, max_features: int | None) -> Pipeline:
    return Pipeline(
        steps=[
            ("vectorizer", build_vectorizer(feature_type, ngram_max, min_df, max_features)),
            ("clf", MultinomialNB(alpha=1.0)),
        ]
    )


def build_lr_pipeline(feature_type: str, ngram_max: int, min_df: int, max_features: int | None) -> Pipeline:
    return Pipeline(
        steps=[
            ("vectorizer", build_vectorizer(feature_type, ngram_max, min_df, max_features)),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    solver="liblinear",
                    class_weight="balanced",
                ),
            ),
        ]
    )


def evaluate_pipeline(name: str, pipeline: Pipeline, X_train, X_test, y_train, y_test) -> ModelResult:
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n" + "=" * 70)
    print(f"Model: {name}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return ModelResult(name=name, accuracy=acc, precision=prec, recall=rec, f1=f1)


def save_pipeline(pipeline: Pipeline, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)


def save_metrics_csv(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def tune_model(
    name: str,
    pipeline: Pipeline,
    param_grid: dict,
    X_train,
    y_train,
    cv: int,
) -> Pipeline:
    """Optional tuning using GridSearchCV (kept simple for MCA)."""
    print("\n" + "-" * 70)
    print(f"Tuning: {name} (cv={cv})")
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)
    return search.best_estimator_


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Fake News Detection models (Naive Bayes & Logistic Regression)."
    )
    parser.add_argument(
        "--features",
        choices=["tfidf", "bow"],
        default="tfidf",
        help="Text features to use: tfidf (default) or bow (Bag of Words).",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=2,
        help="Use n-grams up to this value (default: 2).",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Ignore very rare tokens (minimum document frequency). Default: 2.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=60000,
        help="Limit vocabulary size (default: 60000).",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run simple hyperparameter tuning (slower but often better).",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=3,
        help="Cross-validation folds for tuning (default: 3).",
    )
    args = parser.parse_args()

    dataset = load_kaggle_fake_news_dataset("data/raw")
    print(f"Loaded dataset from: {dataset.source_name}")
    print(f"Total samples: {len(dataset.X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=0.2,
        random_state=42,
        stratify=dataset.y,
    )

    feature_type = args.features
    ngram_max = max(1, int(args.ngram_max))
    min_df = max(1, int(args.min_df))
    max_features = None if args.max_features == 0 else int(args.max_features)

    nb = build_nb_pipeline(feature_type, ngram_max, min_df, max_features)
    lr = build_lr_pipeline(feature_type, ngram_max, min_df, max_features)

    if args.tune:
        # Keep grids small so it is feasible on a student laptop.
        nb = tune_model(
            "Naive Bayes",
            nb,
            param_grid={"clf__alpha": [0.5, 1.0, 2.0]},
            X_train=X_train,
            y_train=y_train,
            cv=args.cv,
        )
        lr = tune_model(
            "Logistic Regression",
            lr,
            param_grid={"clf__C": [0.5, 1.0, 2.0, 4.0]},
            X_train=X_train,
            y_train=y_train,
            cv=args.cv,
        )

    feature_label = "TF-IDF" if feature_type == "tfidf" else "Bag of Words"
    nb_result = evaluate_pipeline(
        f"Naive Bayes ({feature_label})", nb, X_train, X_test, y_train, y_test
    )
    lr_result = evaluate_pipeline(
        f"Logistic Regression ({feature_label})", lr, X_train, X_test, y_train, y_test
    )

    # Save both models so Streamlit can load them
    suffix = "tfidf" if feature_type == "tfidf" else "bow"
    save_pipeline(nb, f"models/naive_bayes_{suffix}.joblib")
    save_pipeline(lr, f"models/logistic_regression_{suffix}.joblib")

    # Save metrics for report writing
    save_metrics_csv(
        rows=[
            {
                "model": nb_result.name,
                "features": feature_label,
                "accuracy": nb_result.accuracy,
                "precision": nb_result.precision,
                "recall": nb_result.recall,
                "f1": nb_result.f1,
            },
            {
                "model": lr_result.name,
                "features": feature_label,
                "accuracy": lr_result.accuracy,
                "precision": lr_result.precision,
                "recall": lr_result.recall,
                "f1": lr_result.f1,
            },
        ],
        path="reports/metrics_latest.csv",
    )
    print("Saved metrics CSV: reports/metrics_latest.csv")

    print("\n" + "=" * 70)
    print("Saved models:")
    print(f"- models/naive_bayes_{suffix}.joblib")
    print(f"- models/logistic_regression_{suffix}.joblib")

    print("\nAccuracy Comparison:")
    print(f"- Naive Bayes        : {nb_result.accuracy:.4f}")
    print(f"- Logistic Regression: {lr_result.accuracy:.4f}")


if __name__ == "__main__":
    main()
