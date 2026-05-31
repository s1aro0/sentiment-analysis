# Movie Review Sentiment Analysis

Binary sentiment classification on the IMDB 50K movie reviews dataset. The project is structured
as three versions, each building on the last — from classic TF-IDF features to fine-tuned transformers.

**Best result:** F1 = 0.9581 with fine-tuned RoBERTa (V3)

---

## Project Versions

| Version | Approach | F1 Score | Error Rate |
|---------|----------|----------|------------|
| V1 | TF-IDF (bigrams) + Linear SVM | 0.9135 | 8.68% |
| V2 | DistilBERT fine-tuned (3 epochs) | 0.9224 | 7.82% |
| V3 | RoBERTa-base fine-tuned (4 epochs, max_len=512) | **0.9581** | **4.22%** |

---

## Dataset

[IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) — Kaggle

- 50,000 reviews (25,000 positive, 25,000 negative)
- Perfectly balanced binary classification task
- Columns: `review` (raw text), `sentiment` (`positive` / `negative`)

---

## Repository Structure

```
Sentiment Analysis/
├── dataset/
│   └── IMDB Dataset.csv               # raw data (not pushed to GitHub)
├── models/
│   ├── v1/                            # Linear SVM + TF-IDF artifacts
│   ├── v2/                            # DistilBERT artifacts
│   └── v3/                            # RoBERTa artifacts (best)
├── sentiment_analysis.ipynb           # V1: TF-IDF + traditional ML
├── sentiment_analysis_v2.ipynb        # V2: DistilBERT fine-tuning
├── sentiment_analysis_v3.ipynb        # V3: RoBERTa fine-tuning
├── predict.py                         # CLI inference script (V3)
├── requirements.txt
└── README.md
```

---

## V1 — TF-IDF + Traditional ML

Full pipeline: EDA, text cleaning, TF-IDF feature extraction, training four models, evaluation,
error analysis, and model saving.

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Linear SVM | 0.9132 | 0.9102 | 0.9168 | 0.9135 |
| Logistic Regression | 0.9076 | 0.9010 | 0.9158 | 0.9084 |
| Multinomial Naive Bayes | 0.8835 | 0.8778 | 0.8910 | 0.8844 |
| Random Forest | 0.8697 | 0.8723 | 0.8662 | 0.8692 |

**TF-IDF config:** bigrams, `max_features=50000`, `sublinear_tf=True`, `min_df=2`

Text preprocessing removes HTML tags, lowercases, and strips stopwords — with negation words
(`not`, `never`, `didn't`, etc.) deliberately kept to preserve sentiment signal.

---

## V2 — DistilBERT Fine-Tuning

Replaces TF-IDF with a fine-tuned `distilbert-base-uncased` transformer (66M parameters).
Training: 3 epochs, `lr=2e-5`, `max_length=256`, batch size 32 on Kaggle T4 GPU (~27 min).

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| DistilBERT | 0.9218 | 0.9155 | 0.9294 | 0.9224 |

---

## V3 — RoBERTa Fine-Tuning

Upgrades to `roberta-base` (125M parameters) with the full 512-token context window.
Training: 4 epochs, `lr=1e-5`, `max_length=512`, batch size 16 on Kaggle T4 GPU (~150 min).
Training loss converged to 0.258 vs 0.411 in V2.

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| RoBERTa | 0.9578 | 0.9524 | 0.9638 | 0.9581 |

Error count dropped from 782 (V2) to 422 (V3) on the 10,000-review test set.

---

## Setup

```bash
pip install -r requirements.txt

# V1 only — download NLTK stopwords
python -c "import nltk; nltk.download('stopwords')"
```

Run the notebooks end-to-end on Kaggle (GPU recommended for V2 and V3).

---

## Inference

Uses the V3 RoBERTa model. Download `models_v3_export.zip` from the Kaggle output panel after
running `sentiment_analysis_v3.ipynb`, extract it into `models/v3/`, then:

```bash
# Single review
python predict.py "This movie was absolutely fantastic."
# Sentiment  : Positive
# Confidence : 0.9970

# Batch — one review per line
python predict.py --file reviews.txt
```

---

## Running on Kaggle

All three notebooks were developed and run on [Kaggle](https://www.kaggle.com) using the following environment:

- **Python** 3.10
- **GPU** Tesla T4 (15.6 GB VRAM)
- **V2 training time:** ~27 minutes
- **V3 training time:** ~150 minutes


---

## Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · NLTK · scikit-learn · Joblib · PyTorch · Hugging Face Transformers

---

