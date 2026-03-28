# Fake News Detector

A machine learning system that classifies news articles as real or fake using Natural Language Processing and TF-IDF vectorization. Built with Python and scikit-learn on a dataset of over 44,000 labeled news articles.

---

## Overview

The spread of misinformation has become one of the defining challenges of the modern information landscape. This project tackles that problem directly — given the text of a news article, can a machine learning model reliably determine whether it is real or fabricated?

The answer, as this project demonstrates, is yes. By combining classical NLP preprocessing techniques with TF-IDF feature extraction and three different classification algorithms, this system achieves an F1 score of 0.99 and an AUC-ROC of 1.00 on held-out test data.

---

## Dataset

- **Source**: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) by Clément Bisaillon (Kaggle)
- **Size**: 44,898 articles total
  - `Fake.csv` — 23,502 fake news articles
  - `True.csv` — 21,417 real news articles
- **Columns**: `title`, `text`, `subject`, `date`
- **Label**: Binary — `fake` or `true` (created manually)

The dataset covers primarily US political news from 2015–2018, sourced from Reuters (real) and various flagged misinformation outlets (fake).

---

## Project Structure

```
fake-news-detector/
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
├── fake_news_model.ipynb       # Main notebook
└── README.md
```

---

## Methodology

### Phase 1 — Data Loading and Exploration

Both CSV files were loaded separately, a binary `label` column was added to each (`fake` / `true`), and the two dataframes were concatenated into a single dataset of 44,898 rows. Class balance was checked to ensure neither class dominated the other significantly.

### Phase 2 — Text Preprocessing

Raw article text is noisy and inconsistent. The following cleaning steps were applied to a combined `content` column (title + article body):

- Converted all text to lowercase
- Removed punctuation using `string.punctuation`
- Removed Unicode artifacts (smart quotes, non-breaking spaces)
- Removed numeric characters
- Removed English stopwords using NLTK (`nltk.corpus.stopwords`)

The goal at this stage is to reduce the vocabulary to meaningful signal words — stripping out everything the model does not need.

### Phase 3 — Exploratory Data Analysis

Three analyses were conducted:

**Most common words in fake news**: Trump, Obama, Clinton, Hillary, video, media, Twitter, America. Heavy on names and emotionally charged terms. Consistent with sensationalist, personality-driven content.

**Most common words in real news**: Said, Reuters, government, state, Washington, republican, united, election. Consistent with formal journalistic language — source attribution and institutional references dominate.

**Article length distribution**: Fake articles tend to be shorter and more variable in length. Real articles cluster around a more consistent length, reflecting editorial standards.

**Subject distribution**: Real news is dominated by `politicsNews` and `worldnews`. Fake news shows a broader spread across categories, with a large volume of general `politics` and `News` labeled content.

### Phase 4 — Feature Engineering (TF-IDF)

Text was converted into a numerical matrix using `TfidfVectorizer` from scikit-learn. TF-IDF (Term Frequency — Inverse Document Frequency) assigns higher weight to words that are frequent within a specific article but rare across the full corpus — effectively capturing the distinctive vocabulary of each article.

The vectorizer was fit exclusively on training data to prevent data leakage. The resulting sparse matrix served as the feature input `X`, with the `label` column as the target `y`. An 80/20 train/test split was applied with `random_state=42` for reproducibility.

### Phase 5 — Model Training

Three classifiers were trained:

| Model | Description |
|---|---|
| Logistic Regression | Linear model; strong baseline for text classification |
| Decision Tree | Non-linear, interpretable; prone to overfitting on text |
| Random Forest | Ensemble of decision trees; more robust than a single tree |

All models were trained on `X_train` and `y_train` only. Test data was never exposed during training.

### Phase 6 — Evaluation

Models were evaluated using classification report, confusion matrix (visualized as a seaborn heatmap), and ROC curve with AUC score.

**Results:**

| Model | F1 Score | AUC-ROC |
|---|---|---|
| Logistic Regression | 0.99 | 1.00 |
| Decision Tree | 1.00 | 1.00 |
| Random Forest | 0.99 | 1.00 |

The Decision Tree's perfect 1.00 F1 score is a red flag — this is likely a sign of overfitting to training patterns. Logistic Regression and Random Forest are the more trustworthy results.

**Confusion matrix analysis** confirmed that misclassification rates were minimal across all three models. The ROC curves for all three hug the top-left corner, indicating near-perfect separation between classes.

**Winner**: Logistic Regression — interpretable, fast, generalizes well, and achieves 0.99 F1 without the overfitting concerns of the Decision Tree.

### Phase 7 — Inference Demo

A prediction function was implemented to classify arbitrary headlines at inference time:

```python
def predict_news(headline):
    headline = headline.lower()
    headline = vectorizer.transform([headline])
    prediction = lr.predict(headline)
    return prediction[0]

predict_news("Trump claims election was stolen")
# Output: 'fake'
```

---

## Results Summary

- F1 Score: **0.99** (Logistic Regression, Random Forest)
- AUC-ROC: **1.00** across all models
- Best model: **Logistic Regression** — best balance of performance and generalization

---

## Key Learnings

- Text classification requires a fundamentally different preprocessing pipeline compared to tabular data — stopword removal, punctuation stripping, and vectorization replace the imputation and scaling steps used in numerical pipelines.
- TF-IDF is a powerful and lightweight feature extraction method that captures word importance relative to the corpus without requiring a neural network.
- High accuracy on this dataset is expected — the vocabulary differences between real and fake news in this corpus are stark enough that even simple models can exploit them.
- A perfect score (1.00) from the Decision Tree should be treated with skepticism, not celebration. Always compare train vs test performance.

---

## Limitations

- The dataset covers a narrow time window (2015–2018) and is heavily US-political in subject matter. The model would likely perform poorly on fake news from other domains or languages.
- The vocabulary gap between Reuters-style real news and the fake news sources in this dataset is unusually large, which inflates performance metrics. Real-world fake news detection is significantly harder.
- The model has no understanding of context, sarcasm, or evolving language — it relies entirely on surface-level word frequency patterns.

---

## Future Work

- **Cross-domain evaluation**: Test the model on news articles from domains outside US politics to measure true generalizability.
- **Deep learning baseline**: Compare TF-IDF + Logistic Regression against a fine-tuned BERT or DistilBERT model to quantify the gap between classical and transformer-based NLP.
- **Feature enrichment**: Incorporate metadata features such as publication source, article length, and subject category alongside TF-IDF.
- **Deployment**: Wrap the model in a FastAPI endpoint and build a minimal frontend where users can paste any article and receive a prediction.
- **Temporal drift**: Evaluate whether a model trained on 2015–2018 data degrades on more recent articles, and explore retraining strategies.

---

## Dependencies

```
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
```

---

## How to Run

1. Clone the repository
2. Download `Fake.csv` and `True.csv` from Kaggle and place them in the `data/` folder
3. Open `fake_news_model.ipynb` in Jupyter or Google Colab
4. Run all cells in order
