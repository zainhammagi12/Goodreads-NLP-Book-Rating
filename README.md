# Goodreads NLP Book Rating Prediction

A natural language processing pipeline that predicts book ratings from Goodreads review text — exploring what language signals correlate with how readers rate books.

---

## Problem Statement

Can we predict whether a Goodreads review will be accompanied by a high or low rating, based solely on the text of the review? This is a classic NLP classification problem with practical applications in recommendation systems, content moderation, and sentiment analysis at scale.

---

## Approach

```
Raw Goodreads Reviews (text + rating)
            │
            ▼
Text Preprocessing
(lowercasing, punctuation removal, stopword removal, lemmatisation)
            │
            ▼
Feature Extraction
(TF-IDF vectorisation, n-gram features)
            │
            ▼
Model Training and Comparison
(Logistic Regression, Naive Bayes, SVM)
            │
            ▼
Evaluation
(accuracy, F1 score, confusion matrix, classification report)
            │
            ▼
Error Analysis
(common misclassification patterns, ambiguous language)
```

---

## Key Findings

- Review sentiment language is a strong predictor of rating band
- Short reviews are harder to classify reliably — less signal for the model
- Certain words appear disproportionately in low-rated reviews regardless of apparent sentiment (e.g. qualified praise)
- Full results and feature importance plots available in the notebook

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3B5526?style=flat)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)

- Python, pandas, NumPy
- NLTK (tokenisation, stopwords, lemmatisation)
- scikit-learn (TF-IDF, classification models, evaluation)
- matplotlib, seaborn (visualisation)
- Jupyter Notebook

---

## Project Structure

```
Goodreads-NLP-Book-Rating/
│
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb    # Data exploration and text cleaning
│   ├── 02_feature_extraction.ipynb       # TF-IDF and n-gram features
│   └── 03_modelling_and_evaluation.ipynb # Model training, comparison, error analysis
│
├── data/
│   └── goodreads_reviews.csv             # Dataset
│
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/zainhammagi12/Goodreads-NLP-Book-Rating

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
jupyter notebook notebooks/
```

---

## Author

**Zain Hammagi** — [linkedin.com/in/zain-hammagi](https://linkedin.com/in/zain-hammagi) · [zainhammagi.github.io](https://zainhammagi.github.io)
