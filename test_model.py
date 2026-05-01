import joblib
from src.text_preprocess import basic_clean_text

# Load the models
lr = joblib.load("models/logistic_regression_tfidf.joblib")
nb = joblib.load("models/naive_bayes_tfidf.joblib")

# Test with real news
real_news = "Simultaneously, the Trinamool held a sit-in protest outside a strongroom at Netaji Indoor Stadium in central Kolkata, where EVMs for several Assembly constituencies in north Kolkata are kept"

cleaned = basic_clean_text(real_news)
print("Cleaned text:", cleaned[:100])
print()

# Test LR
lr_pred = lr.predict([cleaned])[0]
lr_proba = lr.predict_proba([cleaned])[0]
print(f"LR Prediction: {lr_pred}")
print(f"LR Probabilities: {lr_proba}")
print(f"LR Classes: {lr.named_steps['clf'].classes_}")
print()

# Test NB
nb_pred = nb.predict([cleaned])[0]
nb_proba = nb.predict_proba([cleaned])[0]
print(f"NB Prediction: {nb_pred}")
print(f"NB Probabilities: {nb_proba}")
print(f"NB Classes: {nb.named_steps['clf'].classes_}")
