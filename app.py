import os

import joblib
import streamlit as st

from src.text_preprocess import basic_clean_text


@st.cache_resource
def load_models():
    # Streamlit UI stays simple: we load the default TF-IDF models.
    # (Training script also supports BoW via --features bow, if needed.)
    nb = joblib.load("models/naive_bayes_tfidf.joblib")
    lr = joblib.load("models/logistic_regression_tfidf.joblib")
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


def main() -> None:
    st.set_page_config(page_title="Fake News Detection", layout="centered")

    st.title("Fake News Detection")
    st.caption(
        "MCA final year mini/major project (Educational purpose only). "
        "Predictions may be wrong—do not use for real-world decisions."
    )
    st.divider()

    left, right = st.columns([2, 1], gap="large")

    with right:
        st.subheader("Settings")
        algo = st.selectbox(
            "Algorithm",
            ["Naive Bayes (TF-IDF)", "Logistic Regression (TF-IDF)"],
            index=1,
        )
        st.caption(
            "Tip: Logistic Regression usually gives higher accuracy on this dataset."
        )

    with left:
        st.subheader("Input News Text")
        st.caption("Paste headline or full article text.")
        with st.form("predict_form", clear_on_submit=False):
            user_text = st.text_area(
                "News Content",
                height=220,
                placeholder=(
                    "Paste news text here...\n\n"
                    "Example: A new policy was announced today to improve..."
                ),
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Predict")

    # Friendly guidance if models are not trained yet.
    if not (
        os.path.exists("models/naive_bayes_tfidf.joblib")
        and os.path.exists("models/logistic_regression_tfidf.joblib")
    ):
        st.info(
            "Models are not found yet. Please train them first by running:\n\n"
            "`python -m src.train_models --features tfidf`\n\n"
            "Then come back and refresh this page."
        )

    if submitted:
        if not user_text.strip():
            st.warning("Please enter some text to predict.")
            return

        with st.spinner("Analyzing text..."):
            nb, lr = load_models()
            model = nb if algo.startswith("Naive") else lr
            pred, confidence = predict_with_confidence(model, user_text)
            cleaned = basic_clean_text(user_text)

        st.divider()
        st.subheader("Result")

        result_text = label_to_text(pred)
        if pred == 1:
            st.success(f"Prediction: {result_text}")
        else:
            st.error(f"Prediction: {result_text}")

        st.progress(min(max(confidence, 0.0), 1.0), text=f"Confidence: {confidence * 100:.1f}%")

        c1, c2, c3 = st.columns(3)
        c1.metric("Algorithm", "Naive Bayes" if algo.startswith("Naive") else "Logistic Regression")
        c2.metric("Features", "TF-IDF")
        c3.metric("Confidence", f"{confidence * 100:.1f}%")

        raw_words = len(user_text.split())
        cleaned_words = len(cleaned.split())
        st.caption(f"Words: raw={raw_words} | after cleaning={cleaned_words}")

        with st.expander("Show cleaned text (for understanding)"):
            st.code(cleaned if cleaned else "(empty after cleaning)", language="text")


if __name__ == "__main__":
    main()
