"""Streamlit Community Cloud entrypoint.

Streamlit Cloud often expects a file named `streamlit_app.py` by default.
Our main app UI lives in `app.py`, so this file simply delegates to it.
"""

from __future__ import annotations

from pathlib import Path
import io

import joblib
import streamlit as st
import pandas as pd

from src.text_preprocess import (
    basic_clean_text,
    count_urls,
    count_exclamation_marks,
    get_sentiment_polarity,
    get_sentiment_subjectivity,
)
from src.feature_importance import get_top_features, extract_suspicious_words_from_text


# IMPORTANT (Streamlit requirement): set_page_config must be the first Streamlit
# command in the script. If it's called after decorators like @st.cache_resource
# are evaluated, Streamlit Community Cloud can crash on startup.
st.set_page_config(
    page_title="Fake News Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
NB_MODEL_PATH = MODELS_DIR / "naive_bayes_tfidf.joblib"
LR_MODEL_PATH = MODELS_DIR / "logistic_regression_tfidf.joblib"

# Initialize session state
if "clear_text" not in st.session_state:
    st.session_state.clear_text = False


@st.cache_resource
def load_models():
    # Streamlit UI stays simple: we load the default TF-IDF models.
    # (Training script also supports BoW via --features bow, if needed.)
    nb = joblib.load(NB_MODEL_PATH)
    lr = joblib.load(LR_MODEL_PATH)
    return nb, lr


def predict_label(model, text: str) -> int:
    cleaned = basic_clean_text(text)
    return int(model.predict([cleaned])[0])


def predict_with_confidence(model, text: str) -> tuple[int, float]:
    """Returns (predicted_label, confidence).

    Confidence is the predicted probability of the chosen class.
    Works for MultinomialNB and LogisticRegression pipelines.
    """
    cleaned = basic_clean_text(text)
    pred = int(model.predict([cleaned])[0])

    confidence = 0.0
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([cleaned])[0]
        # proba is aligned with model.classes_. For our binary case, we map by class id.
        if hasattr(model, "classes_"):
            class_to_index = {int(c): i for i, c in enumerate(model.classes_)}
            idx = class_to_index.get(pred)
            if idx is not None:
                confidence = float(proba[idx])
        else:
            # Fallback: take max probability
            confidence = float(max(proba))

    return pred, confidence


def label_to_text(label: int) -> str:
    # Assumption used in our training scripts:
    # 0 = Fake News, 1 = Real News
    return "Real News" if label == 1 else "Fake News"


def analyze_text_features(text: str) -> dict:
    """Extract linguistic and style features from text."""
    return {
        "URLs": count_urls(text),
        "Exclamation Marks": count_exclamation_marks(text),
        "Sentiment Polarity": get_sentiment_polarity(text),
        "Subjectivity": get_sentiment_subjectivity(text),
    }


def display_prediction_result(pred: int, confidence: float, user_text: str, model_name: str, model) -> None:
    """Display prediction result with all details."""
    st.subheader("📊 Prediction Result")
    
    col1, col2 = st.columns(2)
    
    with col1:
        result_text = label_to_text(pred)
        if pred == 1:
            st.success(f"✓ **Prediction: {result_text}**")
            st.markdown(f"**Confidence: {confidence * 100:.1f}%**", help="How sure the model is about this prediction")
        else:
            st.error(f"⚠ **Prediction: {result_text}**")
            st.markdown(f"**Confidence: {confidence * 100:.1f}%**", help="How sure the model is about this prediction")
        
        # Confidence bar
        st.progress(min(max(confidence, 0.0), 1.0))
    
    with col2:
        st.metric("Algorithm Used", "Naive Bayes" if "Naive" in model_name else "Logistic Regression")
        st.metric("Feature Type", "TF-IDF")
    
    # Text analysis features
    st.subheader("📝 Text Analysis")
    features_analysis = analyze_text_features(user_text)
    
    feature_cols = st.columns(4)
    for (feature_name, value), col in zip(features_analysis.items(), feature_cols):
        with col:
            if isinstance(value, float):
                st.metric(feature_name, f"{value:.2f}")
            else:
                st.metric(feature_name, value)
    
    # Show important words
    st.subheader("🔍 Important Words (Feature Importance)")
    try:
        top_words, top_scores, interpretation = get_top_features(model, model_name, n_features=15)
        
        if top_words and len(top_words) > 0:
            feature_df = pd.DataFrame({
                "Word": top_words,
                "Importance Score": [f"{abs(s):.4f}" for s in top_scores],
                "Indicates": [interpretation.get(w, "Unknown") for w in top_words]
            })
            st.dataframe(feature_df, width=800, hide_index=True)
            
            # Highlight suspicious words in user text
            suspicious = extract_suspicious_words_from_text(user_text, top_words)
            if suspicious and len(suspicious) > 0:
                st.info(f"**Important words found in your text:** {', '.join(sorted(set(suspicious)))}")
        else:
            st.info("Could not extract feature importance from this model.")
    except Exception as e:
        st.warning(f"⚠️ Could not extract feature importance: {str(e)[:100]}")
    
    # Cleaned text
    cleaned = basic_clean_text(user_text)
    raw_words = len(user_text.split())
    cleaned_words = len(cleaned.split())
    
    with st.expander("📋 Show cleaned text (for understanding)"):
        st.caption(f"Original words: {raw_words} → After cleaning: {cleaned_words}")
        st.code(cleaned if cleaned else "(empty after cleaning)", language="text")


def process_batch_csv(uploaded_file, model_name: str, model) -> pd.DataFrame:
    """Process batch predictions from CSV file."""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check if 'text' column exists
        if 'text' not in df.columns:
            st.error("❌ CSV must have a 'text' column!")
            return None
        
        if len(df) == 0:
            st.error("❌ CSV file is empty!")
            return None
        
        results = []
        progress_bar = st.progress(0)
        
        for idx, row in df.iterrows():
            try:
                text = str(row['text']).strip() if pd.notna(row['text']) else ""
                
                if len(text) == 0:
                    results.append({
                        'text': 'Empty',
                        'prediction': 'SKIPPED',
                        'confidence': 'N/A'
                    })
                else:
                    pred, conf = predict_with_confidence(model, text)
                    results.append({
                        'text': text[:100],
                        'prediction': label_to_text(pred),
                        'confidence': f"{conf * 100:.1f}%"
                    })
            except Exception as e:
                results.append({
                    'text': 'ERROR',
                    'prediction': 'FAILED',
                    'confidence': str(e)[:50]
                })
            
            progress_bar.progress(min((idx + 1) / len(df), 1.0))
        
        results_df = pd.DataFrame(results)
        return results_df
    
    except Exception as e:
        st.error(f"❌ Error processing CSV: {str(e)[:200]}")
        return None


def main() -> None:
    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Choose mode:", ["🏠 Single Prediction", "📊 Batch Processing", "📈 Model Info"])
    
    # Header
    st.title("🚀 Fake News Detection System")
    st.caption(
        "MCA Final Year Project | Educational purpose only | "
        "Do not use for real-world decisions"
    )
    st.divider()
    
    # Check if models exist
    models_exist = NB_MODEL_PATH.exists() and LR_MODEL_PATH.exists()
    if not models_exist:
        st.warning(
            "⚠️ **Models not found!** Please train them first:\n\n"
            "```bash\n"
            "python -m src.train_models --features tfidf --tune\n"
            "```\n\n"
            "Then refresh this page."
        )
        return
    
    # PAGE: Single Prediction
    if page == "🏠 Single Prediction":
        left, right = st.columns([2, 1], gap="large")
        
        with right:
            st.subheader("⚙️ Settings")
            algo = st.radio(
                "Algorithm",
                ["Naive Bayes (TF-IDF)", "Logistic Regression (TF-IDF)"],
                index=1,
            )
            st.caption(
                "💡 **Tip:** Naive Bayes is more reliable for diverse content."
            )
        
        with left:
            st.subheader("📝 Input News Text")
            st.caption("Paste headline or full article text here.")
            
            # Use columns for better layout
            col_input, col_btn = st.columns([4, 1])
            
            with col_input:
                user_text = st.text_area(
                    "News Content",
                    value="" if st.session_state.clear_text else None,
                    height=220,
                    placeholder=(
                        "Example: A new policy was announced today...\n\n"
                        "Or paste any news article here for analysis."
                    ),
                    label_visibility="collapsed",
                )
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                submitted = st.button("🔍 Analyze Text", use_container_width=True, key="analyze_btn")
            with col2:
                if st.button("🗑️ Clear", use_container_width=True, key="clear_btn"):
                    st.session_state.clear_text = True
                    st.rerun()
            with col3:
                st.empty()
            
            # Reset clear flag after rerun
            if st.session_state.clear_text:
                st.session_state.clear_text = False
        
        if submitted:
            if not user_text or not user_text.strip():
                st.warning("⚠️ Please enter some text to predict!")
                return
            
            with st.spinner("🤔 Analyzing text..."):
                nb, lr = load_models()
                model = nb if algo.startswith("Naive") else lr
                pred, confidence = predict_with_confidence(model, user_text)
            
            st.divider()
            display_prediction_result(pred, confidence, user_text, algo, model)
        
        # Example predictions
        with st.expander("📚 See Example Predictions"):
            st.subheader("Sample Real News")
            example_real = "President announces new environmental policy to reduce carbon emissions by 40% over next decade, backed by scientists."
            if st.button("Analyze Example Real News", key="ex_real"):
                with st.spinner("Analyzing..."):
                    nb, lr = load_models()
                    model = nb if algo.startswith("Naive") else lr
                    pred, confidence = predict_with_confidence(model, example_real)
                display_prediction_result(pred, confidence, example_real, algo, model)
            
            st.divider()
            st.subheader("Sample Fake News")
            example_fake = "SHOCKING: Scientists discover that healthy diet is actually deadly poison!!! Click now before they hide this!!"
            if st.button("Analyze Example Fake News", key="ex_fake"):
                with st.spinner("Analyzing..."):
                    nb, lr = load_models()
                    model = nb if algo.startswith("Naive") else lr
                    pred, confidence = predict_with_confidence(model, example_fake)
                display_prediction_result(pred, confidence, example_fake, algo, model)
    
    # PAGE: Batch Processing
    elif page == "📊 Batch Processing":
        st.subheader("📊 Batch News Analysis")
        st.caption("Upload a CSV file with a 'text' column to analyze multiple articles at once.")
        
        algo = st.radio(
            "Algorithm",
            ["Naive Bayes (TF-IDF)", "Logistic Regression (TF-IDF)"],
            index=0,
            key="batch_algo"
        )
        
        uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
        
        if uploaded_file:
            nb, lr = load_models()
            model = nb if algo.startswith("Naive") else lr
            
            if st.button("📊 Process Batch", use_container_width=True):
                with st.spinner("Processing..."):
                    results_df = process_batch_csv(uploaded_file, algo, model)
                
                if results_df is not None:
                    st.success(f"✓ Processed {len(results_df)} articles!")
                    st.dataframe(results_df, width=1000, hide_index=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name="predictions_results.csv",
                        mime="text/csv"
                    )
    
    # PAGE: Model Info
    elif page == "📈 Model Info":
        st.subheader("📈 Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(
                "**Models Used:**\n\n"
                "• Naive Bayes (MultinomialNB)\n"
                "• Logistic Regression\n\n"
                "**Feature Extraction:**\n\n"
                "• TF-IDF Vectorizer\n"
                "• Unigrams & Bigrams\n"
                "• Stop word removal"
            )
        
        with col2:
            st.info(
                "**Training Details:**\n\n"
                "• Dataset: Kaggle Fake News\n"
                "• Train/Test Split: 80/20\n"
                "• Classes: 0=Fake, 1=Real\n\n"
                "**Advanced Features:**\n\n"
                "• URL Detection\n"
                "• Sentiment Analysis\n"
                "• Punctuation Analysis"
            )
        
        st.divider()
        
        # Try to load and display metrics
        metrics_path = ROOT_DIR / "reports" / "metrics_latest.csv"
        if metrics_path.exists():
            st.subheader("📊 Latest Model Metrics")
            metrics_df = pd.read_csv(metrics_path)
            st.dataframe(metrics_df, width=800, hide_index=True)
        else:
            st.info("No metrics file found. Train models to generate metrics.")


if __name__ == "__main__":
    main()

