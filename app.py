"""Streamlit Community Cloud entrypoint.

Streamlit Cloud often expects a file named `streamlit_app.py` by default.
Our main app UI lives in `app.py`, so this file simply delegates to it.
"""

from __future__ import annotations

from pathlib import Path
import io
import json
from datetime import datetime

import joblib
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

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
    page_title="🔍 Advanced Fake News Detection System",
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
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


@st.cache_resource
def load_models():
    # Streamlit UI stays simple: we load the default TF-IDF models.
    # (Training script also supports BoW via --features bow, if needed.)
    nb = joblib.load(NB_MODEL_PATH)
    lr = joblib.load(LR_MODEL_PATH)
    return nb, lr


# ============ ADVANCED UI STYLING ============
def add_custom_css():
    st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 100%);
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #00d4ff;
    }
    .prediction-real {
        background: linear-gradient(135deg, #00ff00 0%, #00aa00 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
    }
    .prediction-fake {
        background: linear-gradient(135deg, #ff0000 0%, #aa0000 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
    }
    .confidence-meter {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        border: 2px solid #00d4ff;
    }
    .advanced-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(0, 212, 255, 0.3);
        margin: 10px 0;
    }
    h1, h2, h3 {
        color: #00ff88;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

add_custom_css()


# ============ ADVANCED ANALYTICS FUNCTIONS ============
def get_risk_level(confidence: float, exclamation_count: int, url_count: int) -> tuple[str, str]:
    """Calculate risk level based on multiple factors"""
    risk_score = 0
    
    # Confidence contribution
    if confidence < 0.6:
        risk_score += 40
    elif confidence < 0.75:
        risk_score += 20
    
    # Exclamation marks
    if exclamation_count > 5:
        risk_score += 30
    elif exclamation_count > 2:
        risk_score += 15
    
    # URLs
    if url_count > 3:
        risk_score += 25
    elif url_count > 0:
        risk_score += 10
    
    if risk_score >= 70:
        return "🔴 CRITICAL", "critical"
    elif risk_score >= 50:
        return "🟠 HIGH", "high"
    elif risk_score >= 25:
        return "🟡 MEDIUM", "medium"
    else:
        return "🟢 LOW", "low"


def create_confidence_gauge(confidence: float, label: str) -> go.Figure:
    """Create a gauge chart for confidence visualization"""
    confidence_pct = confidence * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence_pct,
        title={"text": label},
        domain={"x": [0, 1], "y": [0, 1]},
        number={"suffix": "%", "font": {"size": 40, "color": "#00ff88"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "white"},
            "bar": {"color": "#00ff88", "thickness": 0.7},
            "bgcolor": "rgba(255, 255, 255, 0.05)",
            "steps": [
                {"range": [0, 30], "color": "rgba(255, 0, 0, 0.2)"},
                {"range": [30, 60], "color": "rgba(255, 165, 0, 0.2)"},
                {"range": [60, 100], "color": "rgba(0, 255, 0, 0.2)"}
            ]
        }
    ))
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=14)
    )
    return fig


def create_features_chart(top_words: list, top_scores: list) -> go.Figure:
    """Create a bar chart for important features"""
    fig = go.Figure(go.Bar(
        y=top_words[:10],
        x=[abs(s) for s in top_scores[:10]],
        orientation='h',
        marker=dict(
            color=[s for s in top_scores[:10]],
            colorscale='RdYlGn',
            colorbar=dict(title="Importance")
        )
    ))
    fig.update_layout(
        title="🔍 Top 10 Important Words",
        xaxis_title="Importance Score",
        yaxis_title="Words",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        margin=dict(l=150, r=50, t=50, b=50)
    )
    return fig


def create_analysis_metrics(text: str) -> dict:
    """Extract comprehensive metrics from text"""
    cleaned = basic_clean_text(text)
    
    return {
        "original_length": len(text),
        "cleaned_length": len(cleaned),
        "word_count": len(text.split()),
        "character_count": len([c for c in text if c.isalpha()]),
        "urls": count_urls(text),
        "exclamations": count_exclamation_marks(text),
        "sentiment_polarity": get_sentiment_polarity(text),
        "sentiment_subjectivity": get_sentiment_subjectivity(text),
        "avg_word_length": np.mean([len(w) for w in text.split()]) if text.split() else 0,
    }


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
    """Display advanced prediction result with all details."""
    
    # Get risk level
    exclamations = count_exclamation_marks(user_text)
    urls = count_urls(user_text)
    risk_label, risk_type = get_risk_level(confidence, exclamations, urls)
    
    # Create advanced layout
    st.markdown("---")
    st.markdown("### 📊 **PREDICTION ANALYSIS REPORT**")
    
    # Main prediction with gradient
    col1, col2 = st.columns([2, 1])
    with col1:
        if pred == 1:
            st.markdown(
                f'<div class="prediction-real">✅ REAL NEWS (Authentic Content)</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="prediction-fake">⚠️ FAKE NEWS (Misleading Content)</div>',
                unsafe_allow_html=True
            )
    
    with col2:
        st.markdown(f'<div class="advanced-card"><b>Risk Level:</b><br>{risk_label}</div>', unsafe_allow_html=True)
    
    # Confidence and metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Confidence", f"{confidence * 100:.1f}%", delta=f"±{(1-confidence)*100:.1f}%")
    with col2:
        st.metric("🤖 Model Used", "Naive Bayes" if "Naive" in model_name else "Logistic Regression")
    with col3:
        st.metric("📝 Words Count", len(user_text.split()))
    with col4:
        st.metric("🔗 URLs Found", urls)
    
    # Confidence gauge
    st.markdown("#### Confidence Meter")
    col1, col2 = st.columns([1.5, 1])
    with col1:
        fig_gauge = create_confidence_gauge(confidence, "Model Confidence")
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        st.markdown('<div class="advanced-card">', unsafe_allow_html=True)
        metrics = create_analysis_metrics(user_text)
        st.write(f"**📊 Text Analysis:**")
        st.write(f"- Original Chars: {metrics['original_length']}")
        st.write(f"- Words: {metrics['word_count']}")
        st.write(f"- Avg Word Length: {metrics['avg_word_length']:.1f}")
        st.write(f"- Sentiment: {metrics['sentiment_polarity']:.2f} (−1 to 1)")
        st.write(f"- Subjectivity: {metrics['sentiment_subjectivity']:.2f} (0 to 1)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature importance
    st.markdown("---")
    st.markdown("#### 🔍 **Feature Importance Analysis**")
    try:
        top_words, top_scores, interpretation = get_top_features(model, model_name, n_features=20)
        
        if top_words and len(top_words) > 0:
            # Filter to only show words that appear in the article
            cleaned_text = basic_clean_text(user_text).lower()
            article_words = set(cleaned_text.split())
            
            # Filter top words to only those in the article
            filtered_words = []
            filtered_scores = []
            for word, score in zip(top_words, top_scores):
                if word.lower() in article_words:
                    filtered_words.append(word)
                    filtered_scores.append(score)
                if len(filtered_words) >= 10:
                    break
            
            # Create feature chart with filtered data
            if filtered_words and len(filtered_words) > 0:
                fig_features = create_features_chart(filtered_words, filtered_scores)
                st.plotly_chart(fig_features, use_container_width=True)
            else:
                fig_features = create_features_chart(top_words[:10], top_scores[:10])
                st.plotly_chart(fig_features, use_container_width=True)
            
            # Suspicious words detection
            suspicious = extract_suspicious_words_from_text(user_text, top_words)
            if suspicious and len(suspicious) > 0:
                unique_suspicious = sorted(set(suspicious))
                st.success(f"🚨 **Suspicious Keywords Found:** {', '.join(unique_suspicious)}")
            
            # Feature table with filtered words
            display_words = filtered_words if filtered_words else top_words[:15]
            display_scores = filtered_scores if filtered_words else top_scores[:15]
            st.markdown("**Top Features Breakdown (Words from Article):**")
            feature_df = pd.DataFrame({
                "Word": display_words[:15],
                "Importance": [f"{abs(s):.4f}" for s in display_scores[:15]],
                "Indicates": [interpretation.get(w, "Unknown") for w in display_words[:15]]
            })
            st.dataframe(feature_df, use_container_width=True, hide_index=True)
        else:
            st.info("Could not extract feature importance from this model.")
    except Exception as e:
        st.warning(f"⚠️ Could not extract features: {str(e)[:100]}")
    
    # Cleaned text
    st.markdown("---")
    st.markdown("#### 📋 **Text Processing Details**")
    with st.expander("🔧 Show Preprocessing Info"):
        cleaned = basic_clean_text(user_text)
        raw_words = len(user_text.split())
        cleaned_words = len(cleaned.split())
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Original:** {raw_words} words")
            st.code(user_text[:300] + "..." if len(user_text) > 300 else user_text, language="text")
        with col2:
            st.write(f"**Cleaned:** {cleaned_words} words")
            st.code(cleaned[:300] + "..." if len(cleaned) > 300 else cleaned, language="text")
    
    # Add to history
    st.session_state.prediction_history.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "prediction": "Real News" if pred == 1 else "Fake News",
        "confidence": f"{confidence*100:.1f}%",
        "text_length": len(user_text.split())
    })


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
    # Ensure session state is initialized (defensive initialization)
    if "clear_text" not in st.session_state:
        st.session_state.clear_text = False
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    
    # Advanced sidebar with metrics
    with st.sidebar:
        st.markdown("### 🎛️ **CONTROL PANEL**")
        st.markdown("---")
        
        page = st.radio(
            "**Select Mode:**",
            ["🏠 Single Prediction", "📊 Batch Processing", "📈 Model Analytics", "📜 History"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### 📊 **Session Stats**")
        if st.session_state.prediction_history:
            st.write(f"**Total Predictions:** {len(st.session_state.prediction_history)}")
            real_count = sum(1 for p in st.session_state.prediction_history if "Real" in p['prediction'])
            fake_count = len(st.session_state.prediction_history) - real_count
            st.write(f"🟢 Real: {real_count} | 🔴 Fake: {fake_count}")
        else:
            st.write("No predictions yet.")
    
    # Advanced header
    st.markdown("""
    <div style='text-align: center; margin: 30px 0;'>
    <h1>🔍 Advanced Fake News Detection System</h1>
    <p style='color: #00ff88; font-size: 16px;'>Using AI-Powered Machine Learning | Accuracy: 99.18%</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Check if models exist
    models_exist = NB_MODEL_PATH.exists() and LR_MODEL_PATH.exists()
    if not models_exist:
        st.error(
            "❌ **Models not found!** Please train them first:\n\n"
            "```bash\n"
            "python -m src.train_models --features tfidf --tune\n"
            "```\n\n"
            "Then refresh this page."
        )
        return
    
    # PAGE: Single Prediction
    if page == "🏠 Single Prediction":
        st.markdown("### 📰 **SINGLE NEWS PREDICTION**")
        
        col_settings, col_input = st.columns([1, 2.5], gap="large")
        
        with col_settings:
            st.markdown('<div class="advanced-card">', unsafe_allow_html=True)
            st.subheader("⚙️ Settings")
            
            algo = st.radio(
                "**Select Algorithm:**",
                ["Naive Bayes (TF-IDF)", "Logistic Regression (TF-IDF)"],
                index=0,
            )
            
            st.markdown("---")
            st.write("**Model Accuracy:**")
            if "Naive" in algo:
                st.write("🟢 96.28% Accuracy")
            else:
                st.write("🟢 99.18% Accuracy")
            
            st.markdown("---")
            st.caption("💡 **Tip:** Naive Bayes is more balanced for diverse content.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_input:
            st.markdown('<div class="advanced-card">', unsafe_allow_html=True)
            st.subheader("📝 Input News Content")
            st.caption("Paste any news headline or full article text")
            
            if "text_input_key" not in st.session_state:
                st.session_state.text_input_key = 0
            
            user_text = st.text_area(
                "Enter your news text:",
                height=280,
                placeholder="Example: A new policy was announced today...\n\nOR paste any news article here for AI analysis.",
                label_visibility="collapsed",
                key=f"user_text_{st.session_state.text_input_key}"
            )
            
            col1, col2, col3 = st.columns([1.5, 1, 0.5])
            with col1:
                submitted = st.button("🔍 Analyze Now", use_container_width=True, key="analyze_btn", help="Analyze the text with selected model")
            with col2:
                if st.button("🗑️ Clear", use_container_width=True, key="clear_btn", help="Clear the input text"):
                    st.session_state.text_input_key += 1
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        if submitted:
            if not user_text or not user_text.strip():
                st.error("⚠️ Please enter some text to analyze!")
                return
            
            # Show processing
            with st.spinner("🔄 Running AI analysis... This may take a few seconds"):
                nb, lr = load_models()
                model = nb if algo.startswith("Naive") else lr
                pred, confidence = predict_with_confidence(model, user_text)
            
            # Display result
            display_prediction_result(pred, confidence, user_text, algo, model)
        
        # Example predictions section
        st.markdown("---")
        st.markdown("### 💡 **TRY EXAMPLES**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="advanced-card">', unsafe_allow_html=True)
            st.write("**✅ Example: Authentic News**")
            example_real = "Scientists discover new treatment for cancer through years of research at major medical institutions."
            if st.button("Test This Example", key="ex_real", use_container_width=True):
                with st.spinner("Analyzing..."):
                    nb, lr = load_models()
                    model = nb if algo.startswith("Naive") else lr
                    pred, confidence = predict_with_confidence(model, example_real)
                display_prediction_result(pred, confidence, example_real, algo, model)
            st.caption(example_real[:80] + "...")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="advanced-card">', unsafe_allow_html=True)
            st.write("**🔴 Example: Suspicious Content**")
            example_fake = "SHOCKING!!! Celebrities hate this one weird trick! Click now before they DELETE this!!!"
            if st.button("Test This Example", key="ex_fake", use_container_width=True):
                with st.spinner("Analyzing..."):
                    nb, lr = load_models()
                    model = nb if algo.startswith("Naive") else lr
                    pred, confidence = predict_with_confidence(model, example_fake)
                display_prediction_result(pred, confidence, example_fake, algo, model)
            st.caption(example_fake[:80] + "...")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # PAGE: Batch Processing
    elif page == "📊 Batch Processing":
        st.markdown("### 📊 **BATCH NEWS ANALYSIS**")
        st.caption("Upload a CSV file with a 'text' column to analyze multiple articles at once")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.markdown('<div class="advanced-card">', unsafe_allow_html=True)
            st.subheader("⚙️ Batch Settings")
            algo = st.radio(
                "**Select Algorithm:**",
                ["Naive Bayes (TF-IDF)", "Logistic Regression (TF-IDF)"],
                index=0,
                key="batch_algo"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col1:
            st.markdown('<div class="advanced-card">', unsafe_allow_html=True)
            st.subheader("📁 Upload CSV File")
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=["csv"],
                help="Must contain a 'text' column with news articles",
                label_visibility="collapsed"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file:
            nb, lr = load_models()
            model = nb if algo.startswith("Naive") else lr
            
            if st.button("🚀 Process Batch", use_container_width=True, key="batch_process"):
                with st.spinner("🔄 Processing batch... Analyzing each article with AI"):
                    results_df = process_batch_csv(uploaded_file, algo, model)
                
                if results_df is not None:
                    st.success(f"✅ Successfully processed {len(results_df)} articles!")
                    
                    # Summary stats
                    col1, col2, col3, col4 = st.columns(4)
                    fake_count = sum(results_df['prediction'] == 'Fake')
                    real_count = len(results_df) - fake_count
                    
                    with col1:
                        st.metric("📊 Total Analyzed", len(results_df))
                    with col2:
                        st.metric("🔴 Fake News", fake_count)
                    with col3:
                        st.metric("🟢 Real News", real_count)
                    with col4:
                        avg_conf = results_df['confidence'].astype(float).mean() * 100
                        st.metric("🎯 Avg Confidence", f"{avg_conf:.1f}%")
                    
                    st.markdown("---")
                    st.markdown("#### 📋 **Detailed Results**")
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    # Visualizations
                    st.markdown("---")
                    st.markdown("#### 📈 **Analysis Visualizations**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Prediction pie chart
                        pred_counts = results_df['prediction'].value_counts()
                        fig = go.Figure(data=[go.Pie(
                            labels=pred_counts.index,
                            values=pred_counts.values,
                            marker=dict(colors=['#FF6B6B', '#51CF66'])
                        )])
                        fig.update_layout(title="Prediction Distribution", height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Confidence distribution
                        confidences = results_df['confidence'].astype(float) * 100
                        fig = go.Figure(data=[go.Histogram(
                            x=confidences,
                            nbinsx=20,
                            marker=dict(color='#4C72B0')
                        )])
                        fig.update_layout(title="Confidence Score Distribution", xaxis_title="Confidence %", height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Download results
                    st.markdown("---")
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
    
    # PAGE: Model Analytics
    elif page == "📈 Model Analytics":
        st.markdown("### 📈 **MODEL ANALYTICS & INFORMATION**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="advanced-card">', unsafe_allow_html=True)
            st.subheader("🤖 **Models Used**")
            st.write("• **Naive Bayes** (MultinomialNB)")
            st.write("  - Accuracy: 96.28%")
            st.write("  - Best for: Balanced predictions")
            st.write("")
            st.write("• **Logistic Regression** (Saga solver)")
            st.write("  - Accuracy: 99.18%")
            st.write("  - Best for: High precision")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="advanced-card">', unsafe_allow_html=True)
            st.subheader("🔧 **Feature Engineering**")
            st.write("• **TF-IDF Vectorizer**")
            st.write("• **Max Features:** 60,000")
            st.write("• **N-grams:** Unigrams & Bigrams")
            st.write("• **Stop words:** Removed")
            st.write("• **Preprocessing:** Text cleaning & normalization")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="advanced-card">', unsafe_allow_html=True)
            st.subheader("📊 **Training Details**")
            st.write("• **Dataset:** Kaggle Fake News")
            st.write("• **Total Samples:** 44,898 articles")
            st.write("• **Train/Test Split:** 80/20")
            st.write("• **Feature Vector:** TF-IDF matrix")
            st.write("• **Hyperparameter Tuning:** GridSearchCV")
            st.write("• **Cross-Validation:** 5-fold")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="advanced-card">', unsafe_allow_html=True)
            st.subheader("🎯 **Advanced Analysis**")
            st.write("• **URL Detection:** Identifies links in text")
            st.write("• **Sentiment Analysis:** Polarity & Subjectivity")
            st.write("• **Punctuation Analysis:** Exclamation marks, patterns")
            st.write("• **Risk Assessment:** Multi-factor scoring")
            st.write("• **Feature Importance:** Word-level interpretability")
            st.write("• **Confidence Scoring:** Model probability metrics")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📈 **MODEL PERFORMANCE METRICS**")
        
        metrics_path = ROOT_DIR / "reports" / "metrics_latest.csv"
        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)
            
            # Display metrics as cards
            metrics_data = metrics_df.to_dict('records')[0]
            
            cols = st.columns(len(metrics_data))
            for i, (metric_name, metric_value) in enumerate(metrics_data.items()):
                with cols[i % len(cols)]:
                    try:
                        val = float(metric_value)
                        st.metric(metric_name.replace('_', ' ').title(), f"{val:.2%}" if val < 1 else f"{val:.2f}")
                    except:
                        st.metric(metric_name.replace('_', ' ').title(), metric_value)
            
            st.markdown("---")
            st.markdown("#### 📋 **Full Metrics Table**")
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        else:
            st.info("📊 No metrics file found. Train models to generate performance metrics.")
        
        # Architecture info
        st.markdown("---")
        st.markdown("### 🏗️ **SYSTEM ARCHITECTURE**")
        
        st.code("""
Pipeline Architecture:
1. Input Text → Text Preprocessing (cleaning, normalization)
2. Feature Extraction → TF-IDF Vectorization (60,000 features)
3. Model Selection → Naive Bayes OR Logistic Regression
4. Prediction → Classification + Confidence Score
5. Post-Processing → Risk assessment, analysis metrics
6. Output → Advanced visualization & interpretation
        """, language="text")
    
    # PAGE: History
    elif page == "📜 History":
        st.markdown("### 📜 **PREDICTION HISTORY**")
        
        if st.session_state.prediction_history and len(st.session_state.prediction_history) > 0:
            history_df = pd.DataFrame(st.session_state.prediction_history)
            
            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total Predictions", len(history_df))
            with col2:
                real_count = sum(1 for p in history_df['prediction'] if "Real" in p)
                st.metric("🟢 Real News", real_count)
            with col3:
                fake_count = len(history_df) - real_count
                st.metric("🔴 Fake News", fake_count)
            with col4:
                avg_confidence = history_df['confidence'].str.rstrip('%').astype(float).mean()
                st.metric("🎯 Avg Confidence", f"{avg_confidence:.1f}%")
            
            st.markdown("---")
            st.markdown("#### 📋 **Recent Predictions**")
            st.dataframe(history_df.iloc[::-1], use_container_width=True, hide_index=True)
            
            # Export history
            st.markdown("---")
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="📥 Export History as CSV",
                data=csv,
                file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Clear history
            if st.button("🗑️ Clear History", use_container_width=True, key="clear_history"):
                st.session_state.prediction_history = []
                st.success("✅ History cleared!")
                st.rerun()
        else:
            st.info("📭 No predictions yet. Make some predictions to see them here!")
            st.markdown("**Start with:** 🏠 Single Prediction tab")


if __name__ == "__main__":
    main()

