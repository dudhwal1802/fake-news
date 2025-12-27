# MCA Final Year Project Report (Fake News Detection)

> Educational purpose only: This project is created for learning and academic submission. Do not use predictions for real-world decisions.

## Project Title
Fake News Detection using Machine Learning and NLP (Naive Bayes & Logistic Regression)

## Abstract
Fake news spreads quickly and misleads people. This project detects fake news using Machine Learning and NLP. The dataset is taken from Kaggle. The text is cleaned and converted into TF-IDF features. Two models (Naive Bayes and Logistic Regression) are trained and compared. A Streamlit web interface is built for easy testing.

## Problem Statement
Manual fake news identification is difficult due to large data and fast sharing. An automated classification model is needed to label news as fake or real.

## Objectives
- Use Kaggle dataset for fake news detection.
- Perform text preprocessing.
- Extract features using TF-IDF.
- Train Naive Bayes and Logistic Regression models.
- Compare accuracy and other metrics.
- Provide a Streamlit UI.

## Literature Review (Short)
- Traditional ML approaches with TF-IDF and classifiers (Naive Bayes, Logistic Regression, SVM) are effective for text classification.
- Deep learning approaches can give higher accuracy but need more compute and data.
- For MCA final year projects, classical ML is preferred because it is simple, explainable, and low cost.

## System Architecture
Dataset → Preprocessing → TF-IDF → Model Training (NB + LR) → Evaluation → Save Models → Streamlit UI Prediction

## Data Preprocessing
- Lowercasing
- Removing URLs
- Removing special characters
- Removing extra spaces
- Combining title and text

## Algorithms
### Naive Bayes
Based on Bayes theorem and assumes word independence.

### Logistic Regression
A linear model that predicts class probability using sigmoid function.

## Flowchart (Text)
START → Load Data → Clean Text → TF-IDF → Train/Test Split → Train Models → Evaluate → Save → Streamlit Prediction → END

## Results & Accuracy Comparison
Final results (from latest training run):
| Model | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|----|
| Naive Bayes (TF-IDF) | 0.9614 | 0.9605 | 0.9585 | 0.9595 |
| Logistic Regression (TF-IDF) | 0.9918 | 0.9889 | 0.9939 | 0.9914 |

## Output Screenshots (Description)
1. Streamlit home screen
2. Input text entered
3. Output prediction
4. Training metrics output

## Conclusion
The project successfully classifies news using ML and NLP. Logistic Regression generally performs better, while Naive Bayes is faster and simpler.

## Future Scope
- Use deep learning models
- Add multi-language support
- Add source-based features
- Deploy online

## Viva Voce (Extra Q&A)
1. **Q:** What is NLP?  
	**A:** NLP (Natural Language Processing) is a field of AI that helps computers understand and process human language text.
2. **Q:** What is fake news detection?  
	**A:** It is a classification task where a model predicts whether a news article is fake or real.
3. **Q:** Difference between TF-IDF and Bag of Words (BoW)?  
	**A:** BoW counts how many times words appear, while TF-IDF also reduces the importance of very common words and highlights more informative words.
4. **Q:** Why do we split train and test data?  
	**A:** To evaluate the model on unseen data and check generalization performance.
5. **Q:** What is accuracy?  
	**A:** The percentage of total predictions that are correct.
6. **Q:** What is precision and recall?  
	**A:** Precision measures how many predicted positives are correct; recall measures how many actual positives are detected.
7. **Q:** What is F1-score?  
	**A:** It is the harmonic mean of precision and recall, useful when classes are imbalanced.
8. **Q:** Why Naive Bayes is called “naive”?  
	**A:** Because it assumes all features (words) are independent, which is a simplifying assumption.
9. **Q:** Why Logistic Regression can be used for classification?  
	**A:** It predicts class probabilities using a sigmoid function and assigns a class label based on a threshold.
10. **Q:** What is a confusion matrix?  
	 **A:** A table showing true vs predicted classes (TP, TN, FP, FN) to understand errors.
11. **Q:** Why do we save models using Joblib?  
	 **A:** To reuse trained models without retraining, and to load them inside the Streamlit app.
12. **Q:** What are limitations of this project?  
	 **A:** It depends on the dataset quality; it may not generalize to new writing styles, new topics, or different languages.
