# Fake News Detection (MCA Final Year Project) - ENHANCED VERSION 2.0

**Programming Language:** Python  
**Algorithms:** Naive Bayes, Logistic Regression  
**NLP Feature Extraction:** TF-IDF with Advanced Features  
**Frontend:** Streamlit (Enhanced UI)  
**Dataset:** Kaggle Fake News Dataset  
**Special Features:** Feature Importance Visualization, Sentiment Analysis, Batch Processing, GridSearchCV Tuning

> Educational purpose only: This project is created for learning and academic submission. The output should not be used for real-world decisions.

---

## 1) Project Title
**Fake News Detection using Machine Learning, NLP, and Advanced Feature Engineering**

## 2) Abstract
Fake news spreads quickly on social media and creates misinformation. This project builds a **comprehensive Fake News Detection system** using Machine Learning and Natural Language Processing (NLP). The text is preprocessed and converted into numeric features using TF-IDF along with advanced linguistic features (sentiment analysis, URL detection, punctuation patterns). Two supervised learning algorithms (Naive Bayes and Logistic Regression) are trained, compared, and evaluated with hyperparameter tuning (GridSearchCV). The system includes **feature importance visualization** to explain why a prediction was made, and a **Streamlit web app** with batch processing capability for testing multiple articles. This project is developed using free and open-source tools and is suitable for MCA final year submission.

## Key Improvements in Version 2.0:

### ?? Advanced Backend Features
- **Sentiment Analysis** (TextBlob): Polarity and Subjectivity scores
- **URL Detection**: Count suspicious links (fake news indicator)
- **Punctuation Analysis**: Excessive ! marks (fake news pattern)
- **GridSearchCV**: Automated hyperparameter tuning for optimal models
- **Feature Importance**: Shows top 15 words influencing each prediction
- **Better Metrics**: Comprehensive comparison table with best model recommendation

### ?? Enhanced UI Features  
- **3 Modes**: Single Prediction, Batch Processing, Model Info Dashboard
- **Word Highlighting**: Highlights important words in your input text
- **Text Analytics**: Shows URLs, sentiment, punctuation counts
- **Batch CSV Processing**: Analyze 100+ articles in seconds
- **Example Predictions**: Built-in samples to test quickly
- **Professional Dashboard**: Metrics, training details, algorithm info

### ?? Better Model Comparison
- Side-by-side accuracy, precision, recall, F1-score
- Best model recommendation
- Saved metrics CSV for your report
- Confusion matrices and classification reports

---

## Quick Start (5 minutes)

### 1. Install Requirements
`ash
pip install -r requirements.txt
`

### 2. Train Models (with Tuning)
`ash
python -m src.train_models --features tfidf --tune --cv 5
`

### 3. Run Streamlit App
`ash
streamlit run app.py
`

That's it! ??

---

## File Structure
`
fake-news/
  +-- app.py (UPDATED - Enhanced UI)
  +-- requirements.txt (UPDATED - New packages)
  +-- README.md (THIS FILE - Comprehensive guide)
  +-- data/raw/ (Your CSV files)
  +-- models/ (Trained models)
  +-- src/
  �   +-- text_preprocess.py (UPDATED - New features)
  �   +-- train_models.py (UPDATED - Better GridSearchCV)
  �   +-- feature_importance.py (NEW - Visualization)
  �   +-- data_loader.py
  +-- reports/ (metrics_latest.csv)
`

---

## Detailed Sections

### NEW Advanced Features

#### 1. Sentiment Analysis
- **What**: Analyzes if text is positive/negative/neutral
- **Why**: Fake news uses extreme emotions
- **How**: TextBlob polarity (-1 to +1) and subjectivity (0 to 1)
- **Insight**: Highly subjective + extreme sentiment = likely fake

#### 2. URL Detection  
- **What**: Counts number of suspicious links
- **Why**: Fake news contains more malicious links
- **How**: Regex pattern matching http:// and www.
- **Insight**: High URL count + short article = suspicious

#### 3. Punctuation Analysis
- **What**: Counts excessive !, ? marks
- **Why**: Fake news uses sensationalism
- **How**: Count special punctuation characters
- **Insight**: Excessive ! marks = emotional manipulation signal

#### 4. GridSearchCV Hyperparameter Tuning
- **What**: Automatically finds best model parameters
- **Why**: Better accuracy than default parameters
- **Parameters Tested**:
  - Naive Bayes: alpha [0.5, 1.0, 2.0]
  - Logistic Regression: C [0.5, 1.0, 2.0, 4.0]
- **How**: Cross-validation (CV=5) for robust evaluation

#### 5. Feature Importance Visualization
- **What**: Shows top 15 words influencing the prediction
- **For Logistic Regression**: Coefficient magnitude
  - High positive = REAL news indicator
  - High negative = FAKE news indicator
- **For Naive Bayes**: Log probability differences
- **UI Display**: Table + highlighted words in input text

### Enhanced Streamlit UI

#### Mode 1: ?? Single Prediction
- Paste news text
- Choose algorithm (Naive Bayes or Logistic Regression)
- See:
  - Prediction (Fake/Real)
  - Confidence percentage
  - Text analytics (URLs, sentiment, punctuation)
  - **Feature importance (Top 15 words)**
  - **Highlighted important words in your text**
  - Cleaned text view

#### Mode 2: ?? Batch Processing
- Upload CSV with 'text' column
- Instantly process 100+ articles
- Get results table with predictions & confidence
- Download as CSV

#### Mode 3: ?? Model Info
- Show training metrics
- Model comparison table
- Algorithm & feature details

---

## Training with Improvements

### Command
`ash
# Basic training
python -m src.train_models --features tfidf

# WITH HYPERPARAMETER TUNING (Recommended)
python -m src.train_models --features tfidf --tune --cv 5

# Advanced
python -m src.train_models --features tfidf --tune --cv 5 --ngram-max 2 --max-features 50000
`

### Output
- ? Model comparison table
- ? Best model recommendation  
- ? Saved models (.joblib)
- ? Metrics CSV for reports
- ? Confusion matrices
- ? Classification reports

Example:
`
======================================================================
?? MODEL COMPARISON RESULTS
======================================================================
                                Model    Accuracy  Precision  Recall  F1
Naive Bayes (TF-IDF)                    0.9614    0.9605     0.9585  0.9595
Logistic Regression (TF-IDF)            0.9918    0.9889     0.9939  0.9914

? BEST MODEL: Logistic Regression (TF-IDF)
  Accuracy: 0.9918
`

---

## Viva Voce Preparation (15 Key Questions)

### Machine Learning Basics
1. **Q: What is fake news detection?**
   - A: Classifying news as fake/real using ML and providing explainability.

2. **Q: How does Naive Bayes work?**
   - A: Probabilistic classifier using Bayes' theorem. Assumes feature independence.

3. **Q: How does Logistic Regression work?**
   - A: Linear classifier using sigmoid function for probability outputs.

4. **Q: Why use both NB and LR?**
   - A: Compare performance, understand trade-offs, demonstrate model selection methodology.

### Text Processing & Features
5. **Q: Why is text preprocessing needed?**
   - A: Remove noise, normalize text, extract meaningful features for better performance.

6. **Q: What is TF-IDF and why use it?**
   - A: Measures term importance (frequency � uniqueness). Captures both dimensions of word significance.

7. **Q: What are the NEW preprocessing features?**
   - A: Sentiment analysis (TextBlob), URL detection, punctuation analysis.

8. **Q: Why add sentiment analysis?**
   - A: Fake news uses extreme emotions. High sentiment is a red flag.

### Model Improvements
9. **Q: What is GridSearchCV?**
   - A: Automatically tests parameter combinations via cross-validation to find best model.

10. **Q: What is feature importance?**
    - A: Shows which input features (words) most strongly influence predictions.

11. **Q: How to extract feature importance?**
    - A: For LR: magnitude of coefficients. For NB: log probability differences.

### Evaluation & Explainability  
12. **Q: Difference between precision and recall?**
    - A: Precision="of predicted positives, how many correct?" Recall="of actual positives, how many found?"

13. **Q: How does the system explain predictions?**
    - A: Shows top 15 important words, highlights them in text, displays metrics.

### Project Specific
14. **Q: What makes this suitable for MCA final year?**
    - A: Demonstrates ML fundamentals, NLP, feature engineering, UI development, and explainability.

15. **Q: What improvements were made in v2.0?**
    - A: Sentiment analysis, URL detection, GridSearchCV tuning, feature importance visualization, batch processing.

---

## Installation & Dependencies

### New Packages in v2.0
- 	extblob - Sentiment analysis
- matplotlib - Visualization
- plotly - Interactive charts
- openpyxl - Excel support

### Install All
`ash
pip install -r requirements.txt
`

---

## References
- Kaggle Dataset: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- Scikit-learn: https://scikit-learn.org/stable/
- Streamlit Docs: https://docs.streamlit.io/
- TextBlob Docs: https://textblob.readthedocs.io/
- Pandas Docs: https://pandas.pydata.org/docs/

---

## Tips for Presentation
1. **Show feature importance**: Very impressive to examiners
2. **Demonstrate batch processing**: Shows understanding of scalability
3. **Explain why sentiment matters**: Show examples of extreme sentiment in fake news
4. **Talk about GridSearchCV**: Shows knowledge of hyperparameter tuning
5. **Highlight explainability**: Modern AI concept that impresses

---

**Version 2.0 Enhanced | March 2026 | MCA Final Year Project**
