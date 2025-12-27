from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import pandas as pd

from src.text_preprocess import basic_clean_text, combine_title_and_text


@dataclass
class LoadedDataset:
    X: pd.Series
    y: pd.Series
    source_name: str


def _ensure_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "Place Kaggle dataset files inside data/raw/ (see README)."
        )


def load_kaggle_fake_news_dataset(data_dir: str = "data/raw") -> LoadedDataset:
    """Loads Kaggle fake news dataset.

    Supports two common Kaggle formats:

    A) 'Fake and Real News' dataset:
       - Fake.csv (label=0)
       - True.csv (label=1)
       Columns usually: title, text, subject, date

    B) 'Fake News' dataset (often train.csv):
       - train.csv with label column (0=fake, 1=real)
       Columns usually: title, author, text, label

    Returns:
        LoadedDataset with cleaned combined text and labels.
    """
    fake_csv = os.path.join(data_dir, "Fake.csv")
    true_csv = os.path.join(data_dir, "True.csv")
    train_csv = os.path.join(data_dir, "train.csv")
    custom_csv = os.path.join(data_dir, "custom_labeled.csv")

    if os.path.exists(fake_csv) and os.path.exists(true_csv):
        fake_df = pd.read_csv(fake_csv)
        true_df = pd.read_csv(true_csv)

        fake_df["label"] = 0
        true_df["label"] = 1

        df = pd.concat([fake_df, true_df], ignore_index=True)

        # Optional: merge user-added recent labeled samples for better coverage.
        # Format: custom_labeled.csv with columns: text,label (0=fake, 1=real)
        if os.path.exists(custom_csv):
            extra = pd.read_csv(custom_csv)
            if not {"text", "label"}.issubset(set(extra.columns)):
                raise ValueError(
                    "custom_labeled.csv must contain columns: text,label (0=fake, 1=real)."
                )
            extra = extra[["text", "label"]].copy()
            df = pd.concat([df, extra], ignore_index=True)
        title = df.get("title")
        text = df.get("text")
        # If extra rows don't have 'title', title will be NaN/None; that's okay.
        combined = [combine_title_and_text(t, b) for t, b in zip(title, text)]
        X = pd.Series([basic_clean_text(x) for x in combined])
        y = df["label"].astype(int)
        source_name = "Fake.csv + True.csv"
        if os.path.exists(custom_csv):
            source_name += " + custom_labeled.csv"
        return LoadedDataset(X=X, y=y, source_name=source_name)

    if os.path.exists(train_csv):
        df = pd.read_csv(train_csv)
        if "label" not in df.columns:
            raise ValueError("train.csv found but 'label' column is missing.")

        if os.path.exists(custom_csv):
            extra = pd.read_csv(custom_csv)
            if not {"text", "label"}.issubset(set(extra.columns)):
                raise ValueError(
                    "custom_labeled.csv must contain columns: text,label (0=fake, 1=real)."
                )
            extra = extra[["text", "label"]].copy()
            df = pd.concat([df, extra], ignore_index=True)

        title = df["title"] if "title" in df.columns else ""
        text = df["text"] if "text" in df.columns else df.iloc[:, 0]
        combined = [combine_title_and_text(t, b) for t, b in zip(title, text)]
        X = pd.Series([basic_clean_text(x) for x in combined])
        y = df["label"].astype(int)
        source_name = "train.csv"
        if os.path.exists(custom_csv):
            source_name += " + custom_labeled.csv"
        return LoadedDataset(X=X, y=y, source_name=source_name)

    # If nothing found, raise with clear message
    _ensure_exists(fake_csv)  # will raise with guidance
    raise RuntimeError("Unreachable")
