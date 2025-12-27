import re
from typing import Optional


def basic_clean_text(text: str) -> str:
    """Basic, exam-friendly text cleaning.

    Keeps it simple and explainable (MCA-level):
    - Lowercase
    - Remove URLs
    - Remove non-letters
    - Collapse extra spaces
    """
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def combine_title_and_text(title: Optional[str], body: Optional[str]) -> str:
    title = "" if title is None else str(title)
    body = "" if body is None else str(body)
    combined = f"{title} {body}".strip()
    return combined
