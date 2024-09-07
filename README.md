# Goodreads-NLP-Book-Rating

This project aims to predict book ratings on Goodreads based on user review text using machine learning and deep learning models. It was completed as part of the *CS985: Machine Learning for Data Analytics* coursework.

## Project Overview

The goal of this project was to predict Goodreads review ratings (1-5) using text reviews. Several models were implemented and compared:
- **Naive Bayes**
- **3-Layer Neural Network**
- **1D CNN with GloVe Embeddings**
- **LSTM**
- **GRU**

The **1D CNN** model with GloVe embeddings achieved the highest accuracy.

### Dataset
The dataset used for this task consists of 900,000 reviews from Goodreads, including features such as:
- `review_text`: The text of the review (used for prediction).
- `rating`: The rating of the book (target variable).

### Key Steps
1. **Data Preprocessing**: Text cleaning, tokenization, and vectorization using TF-IDF.
2. **Modeling**: Training several machine learning models, including deep learning architectures like CNNs and LSTMs.
3. **Evaluation**: The models were evaluated based on accuracy, with the 1D CNN model performing the best.

## Files in This Repository

- `notebooks/Goodreads.ipynb`: The Jupyter notebook containing all code for data preprocessing, model training, and evaluation.
- `docs/Goodreads-Doc.pdf`: The final report detailing the project, methodology, and results.

## How to Run

To reproduce the results:
1. Clone the repository:
   ```bash
   git clone https://github.com/zainhammagi12/Goodreads-NLP-Book-Rating.git
   cd Goodreads-NLP-Book-Rating
   
## Kaggle Competition

Link: https://www.kaggle.com/competitions/cs985-cs987-goodread-class-project
