"""Feature Importance Analysis and Visualization.

This module extracts the most important features (words) from trained models
and provides functions to display them in the Streamlit UI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def get_top_features_logistic_regression(
    pipeline: Pipeline, n_features: int = 20
) -> tuple[list[str], list[float]]:
    """Extract top important words from Logistic Regression model.
    
    For LR, importance is based on the absolute value of model coefficients.
    High positive = indicates REAL news
    High negative = indicates FAKE news
    
    Args:
        pipeline: Trained Pipeline with TfidfVectorizer and LogisticRegression
        n_features: Number of top features to return (default: 20)
    
    Returns:
        Tuple of (feature_names, importance_scores)
    """
    vectorizer = pipeline.named_steps.get("vectorizer")
    model = pipeline.named_steps.get("clf")
    
    if vectorizer is None or model is None:
        return [], []
    
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    
    # Get indices of top important features (by absolute value)
    top_indices = np.argsort(np.abs(coefficients))[-n_features:][::-1]
    
    top_features = [feature_names[i] for i in top_indices]
    top_scores = [coefficients[i] for i in top_indices]
    
    return top_features, top_scores


def get_top_features_naive_bayes(
    pipeline: Pipeline, n_features: int = 20
) -> tuple[list[str], list[float]]:
    """Extract top important words from Naive Bayes model.
    
    For NB, importance is based on log probability differences.
    
    Args:
        pipeline: Trained Pipeline with TfidfVectorizer and MultinomialNB
        n_features: Number of top features to return (default: 20)
    
    Returns:
        Tuple of (feature_names, importance_scores)
    """
    vectorizer = pipeline.named_steps.get("vectorizer")
    model = pipeline.named_steps.get("clf")
    
    if vectorizer is None or model is None:
        return [], []
    
    feature_names = vectorizer.get_feature_names_out()
    
    # For NB, feature importance can be derived from feature_log_prob_
    # Difference between prob of class 1 (real) and class 0 (fake)
    if hasattr(model, "feature_log_prob_"):
        importance = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    else:
        return [], []
    
    top_indices = np.argsort(np.abs(importance))[-n_features:][::-1]
    
    top_features = [feature_names[i] for i in top_indices]
    top_scores = [importance[i] for i in top_indices]
    
    return top_features, top_scores


def get_top_features(
    pipeline: Pipeline, model_name: str, n_features: int = 20
) -> tuple[list[str], list[float], dict]:
    """Get top important features from any trained model.
    
    Args:
        pipeline: Trained Pipeline
        model_name: Name of model ("Naive Bayes" or "Logistic Regression")
        n_features: Number of top features to return
    
    Returns:
        Tuple of (feature_names, importance_scores, interpretation_dict)
    
    Raises:
        ValueError: If pipeline structure is invalid
    """
    if pipeline is None:
        raise ValueError("Pipeline is None")
    
    if "Naive Bayes" in model_name or "naive" in model_name.lower():
        features, scores = get_top_features_naive_bayes(pipeline, n_features)
    else:
        features, scores = get_top_features_logistic_regression(pipeline, n_features)
    
    if not features or len(features) == 0:
        return [], [], {}
    
    # Create interpretation for each feature
    interpretation = {}
    for feature, score in zip(features, scores):
        if score > 0:
            interpretation[feature] = "→ Indicates REAL News"
        else:
            interpretation[feature] = "→ Indicates FAKE News"
    
    return features, scores, interpretation


def create_feature_importance_dataframe(
    features: list[str], scores: list[float]
) -> pd.DataFrame:
    """Create a DataFrame for displaying feature importance.
    
    Args:
        features: List of feature names
        scores: List of importance scores
    
    Returns:
        DataFrame with features and their importance
    """
    df = pd.DataFrame({
        "Word/Feature": features,
        "Importance Score": scores,
        "Direction": ["REAL ↑" if s > 0 else "FAKE ↓" for s in scores]
    })
    return df


def extract_suspicious_words_from_text(
    text: str, features: list[str]
) -> list[str]:
    """Find which words from the text are in the important features list.
    
    Args:
        text: Input text to analyze
        features: List of important feature words
    
    Returns:
        List of words from text that match important features
    """
    words = text.lower().split()
    important_words_found = [w for w in words if w in features]
    return important_words_found
