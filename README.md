# Fake News Detection (MCA Final Year Project)

**Programming Language:** Python  
**Algorithms:** Naive Bayes, Logistic Regression  
**NLP Feature Extraction:** TF-IDF (Bag of Words can be added easily)  
**Frontend:** Streamlit  
**Dataset:** Kaggle Fake News Dataset  

> Educational purpose only: This project is created for learning and academic submission. The output should not be used for real-world decisions.

---

## 1) Project Title
**Fake News Detection using Machine Learning and NLP (Naive Bayes & Logistic Regression)**

## 2) Abstract
Fake news spreads quickly on social media and creates misinformation. This project builds a simple Fake News Detection system using Machine Learning and Natural Language Processing (NLP). The text is cleaned and converted into numeric features using TF-IDF. Two supervised learning algorithms (Naive Bayes and Logistic Regression) are trained and compared. A Streamlit web app is created to allow users to paste news text and get a prediction as Fake or Real. This project is developed using free and open-source tools and is suitable for MCA final year submission.

## 3) Problem Statement
Manually identifying fake news is difficult because of large volume, fast sharing, and similar writing styles. We need an automated system that can classify a news article as **Fake** or **Real** using text analysis.

## 4) Objectives
- Collect and use the Kaggle Fake News Dataset.
- Clean and preprocess news text for ML.
- Convert text into features using TF-IDF.
- Train and evaluate Naive Bayes and Logistic Regression models.
- Compare results and select the better model.
- Build a Streamlit UI for easy testing.

## 5) Literature Review (Short)
- Early fake news detection used manual rule-based approaches (keyword patterns) but these methods do not generalize well.
- Traditional ML models such as Naive Bayes and Logistic Regression work well for text classification with Bag-of-Words/TF-IDF features.
- Recent research uses deep learning (LSTM, Transformers), but they require more compute and larger data. For MCA-level projects, classical ML models are accurate, explainable, and easy to implement.

## 6) System Architecture (Textual)
**Architecture Components:**
1. **Dataset (Kaggle)** → news text + labels
2. **Preprocessing** → cleaning (lowercase, remove URLs, remove symbols)
3. **Feature Extraction (TF-IDF)** → converts text to numeric vectors
4. **Model Training** → Naive Bayes, Logistic Regression
5. **Evaluation** → accuracy, precision, recall, F1-score
6. **Model Saving** → Joblib saved models
7. **Streamlit App** → user enters text → model predicts Fake/Real

**Explanation:**
The system takes labeled news text from Kaggle. The text is cleaned and transformed into TF-IDF features. Two ML models are trained and tested. The models are saved and later loaded in the Streamlit application to provide real-time predictions.

## 7) Data Preprocessing Steps
1. Load CSV dataset from `data/raw/`.
2. Combine title and text (if both are present).
3. Convert to lowercase.
4. Remove URLs and special characters.
5. Remove extra spaces.
6. Split dataset into train/test sets.

Optional (for better accuracy on today's news):
- Add your own latest labeled samples in `data/raw/custom_labeled.csv`
- Columns must be: `text,label` where `label` is `0=fake` and `1=real`
- Retrain the model again

## 8) Algorithm Explanation
### A) Naive Bayes (MultinomialNB)
- Naive Bayes is a probabilistic classifier based on Bayes’ theorem.
- It assumes features (words) are conditionally independent.
- Works fast and performs well on text classification problems.

### B) Logistic Regression
- Logistic Regression is a linear classifier.
- It learns weights for each feature to separate Fake vs Real classes.
- Often performs strongly for TF-IDF text classification.

## 9) Flowchart (Textual)
START
→ Load Kaggle Dataset
→ Clean Text (lowercase, remove URLs, remove symbols)
→ Convert Text to TF-IDF Features
→ Split Train/Test
→ Train Naive Bayes Model
→ Train Logistic Regression Model
→ Evaluate Both Models
→ Save Best/All Models
→ Streamlit UI loads models
→ User inputs text
→ Predict Fake/Real
END

## 10) How to Run (VS Code)
### Step 1: Put dataset files
Download Kaggle dataset and place the CSVs in:
- `data/raw/`

Supported formats:
- `Fake.csv` and `True.csv` (Fake=0, Real=1)
- OR `train.csv` (must contain `label` column)

Optional (recommended for new/real-time news):
- Create `data/raw/custom_labeled.csv` with columns `text,label`
- Add latest real/fake examples (from trusted sources / fact-check websites)
- Retrain so the model learns new topics and writing styles

### Step 2: Install requirements
```bash
pip install -r requirements.txt
```

### Step 3: Train models
```bash
python -m src.train_models --features tfidf
```
This creates:
- `models/naive_bayes_tfidf.joblib`
- `models/logistic_regression_tfidf.joblib`

Optional (Bag of Words):
```bash
python -m src.train_models --features bow
```
This creates:
- `models/naive_bayes_bow.joblib`
- `models/logistic_regression_bow.joblib`

### Step 4: Run Streamlit app
```bash
streamlit run app.py
```

## 11) Output Screenshots (What to Capture)
- Screenshot 1: Streamlit home page with input box.
- Screenshot 2: Entered sample news text.
- Screenshot 3: Prediction output (badge + confidence %) for Naive Bayes.
- Screenshot 4: Prediction output (badge + confidence %) for Logistic Regression.
- Screenshot 5: Training terminal output showing accuracy and confusion matrix.

(Place screenshots inside `screenshots/` folder.)

## 12) Result & Accuracy Comparison
After training, you will get metrics in the terminal.
Record them in your report.

Example format:
| Model | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|----|
| Naive Bayes (TF-IDF) | (your value) | (your value) | (your value) | (your value) |
| Logistic Regression (TF-IDF) | (your value) | (your value) | (your value) | (your value) |

## 13) Conclusion
This project shows that classical ML algorithms can detect fake news effectively using TF-IDF features. Logistic Regression often provides higher accuracy, while Naive Bayes is faster and simpler. The Streamlit app makes the system easy to use.

## 14) Future Scope
- Use advanced NLP models like Word2Vec, FastText, or Transformers.
- Add language detection and multi-language support.
- Use more features (source credibility, author, publication date).
- Deploy on cloud (free tier) for public access.

## 15) Viva Voce Questions & Answers (10)
1. **Q:** What is fake news detection?  
   **A:** It is the task of classifying news as fake or real using computational methods.
2. **Q:** Why is text preprocessing needed?  
   **A:** To remove noise and make the text consistent for better ML performance.
3. **Q:** What is TF-IDF?  
   **A:** It measures how important a word is in a document compared to the whole dataset.
4. **Q:** What is Bag of Words?  
   **A:** A technique that represents text as word frequency counts.
5. **Q:** Why Naive Bayes is called “naive”?  
   **A:** Because it assumes all features are independent, which is rarely fully true.
6. **Q:** Why Logistic Regression is used for classification?  
   **A:** It outputs probabilities and can separate classes using a linear decision boundary.
7. **Q:** What is overfitting?  
   **A:** When a model performs well on training data but poorly on unseen test data.
8. **Q:** What is a confusion matrix?  
   **A:** A table that shows correct and incorrect predictions for each class.
9. **Q:** What is precision and recall?  
   **A:** Precision measures correctness of positive predictions; recall measures coverage of actual positives.
10. **Q:** How do you deploy this project?  
   **A:** Train and save the model, then run the Streamlit app locally or on a hosting platform.

---

## GitHub Folder Structure
```
fake-news/
  app.py
  requirements.txt
  README.md
  .gitignore
  data/
    raw/
    processed/
  models/
  src/
    __init__.py
    data_loader.py
    text_preprocess.py
    train_models.py
  reports/
  screenshots/
```
