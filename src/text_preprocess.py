import re
from typing import Optional
from textblob import TextBlob


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


def count_urls(text: str) -> int:
    """Count number of URLs/links in the text."""
    if text is None:
        return 0
    urls = re.findall(r"http\S+|www\S+", str(text))
    return len(urls)


def count_uppercase_ratio(text: str) -> float:
    """Returns ratio of uppercase letters (0.0 to 1.0)."""
    if text is None or len(text) == 0:
        return 0.0
    text = str(text)
    uppercase = sum(1 for c in text if c.isupper())
    return uppercase / len(text) if len(text) > 0 else 0.0


def count_exclamation_marks(text: str) -> int:
    """Count exclamation marks (often used in fake news)."""
    if text is None:
        return 0
    return str(text).count("!")


def count_question_marks(text: str) -> int:
    """Count question marks."""
    if text is None:
        return 0
    return str(text).count("?")


def get_sentiment_polarity(text: str) -> float:
    """Returns sentiment polarity score (-1.0 to 1.0).
    
    -1.0: Very negative
     0.0: Neutral
     1.0: Very positive
    """
    if text is None or len(text) == 0:
        return 0.0
    try:
        blob = TextBlob(str(text))
        return blob.sentiment.polarity
    except Exception:
        return 0.0


def get_sentiment_subjectivity(text: str) -> float:
    """Returns sentiment subjectivity score (0.0 to 1.0).
    
    0.0: Objective (factual)
    1.0: Subjective (opinionated)
    Fake news tends to be more subjective.
    """
    if text is None or len(text) == 0:
        return 0.0
    try:
        blob = TextBlob(str(text))
        return blob.sentiment.subjectivity
    except Exception:
        return 0.0


def combine_title_and_text(title: Optional[str], body: Optional[str]) -> str:
    title = "" if title is None else str(title)
    body = "" if body is None else str(body)
    combined = f"{title} {body}".strip()
    return combined
