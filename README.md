# Fake vs Real News: Exploratory Data Analysis

## Project definition
An exploratory data analysis (EDA) of ~45k news articles labelled **Real** (`True.csv`, Reuters-sourced) or **Fake** (`Fake.csv`, flagged unreliable sites). The goal is to find out what separates the two classes (size, topic, length, timing, vocabulary) and which of those differences are genuine signals vs. dataset artefacts, *before* any model is built.

## Dataset
| File | Rows | Columns |
|------|------|---------|
| True.csv | 21,417 | title, text, subject, date |
| Fake.csv | 23,481 | title, text, subject, date |

After removing exact duplicates (5,793) and empty-text rows: **21,196 Real** and **17,462 Fake** articles.

### Use cases
- Baseline understanding before building a fake-news classifier
- Teaching data cleaning + EDA on text data
- Spotting dataset leakage (features that give away the label without real signal)
- Media-literacy research on writing patterns of misleading content

## Tools
Python 3, pandas, NumPy, Matplotlib (+ `re` and `collections.Counter` from the standard library). No other libraries.

## The 5 graphs and what they show
| # | Graph | Outcome |
|---|-------|---------|
| 1 | Class balance (bar) | Fairly balanced, Real is a bit larger after de-duplication |
| 2 | Articles per subject by label | The subject columns **do not overlap**: Real = politicsNews/worldnews, Fake = News/politics/left-news/etc. So `subject` leaks the label and must not be used as a model feature |
| 3 | Article length histogram | Medians are close (Real 359 words, Fake 376), so length alone is a weak signal |
| 4 | Articles per month (line) | The two classes cover different time spans and volumes, another artefact to be careful about |
| 5 | Top 15 words per class | Real: 'reuters', 'told', 'government', 'state', 'washington' (formal reporting vocabulary). Fake: 'donald', 'him', 'our', 'like', 'because', 'even' (personal, emotional, opinionated tone) |

**Extra finding:** ~99% of Real articles contain "Reuters" in the first 120 characters vs ~0.1% of Fake ones. A classifier would just learn that tag. Strip it before modelling.

## How to run in Google Colab
1. Open https://colab.research.google.com -> **File -> Upload notebook** -> choose `fake_news_eda.ipynb` (or New notebook and paste `fake_news_eda.py` cell by cell).
2. Click the **folder icon** on the left, then **Upload to session storage**, and upload `True.csv` and `Fake.csv`.
3. **Runtime -> Run all.** Plots appear inline and PNGs are saved in the session.

Alternative: mount Google Drive, then set
```python
from google.colab import drive
drive.mount('/content/drive')
TRUE_PATH = "/content/drive/MyDrive/fake-news/True.csv"
FAKE_PATH = "/content/drive/MyDrive/fake-news/Fake.csv"
```

## Repo structure
```
fake-news-eda/
├── README.md
├── fake_news_eda.ipynb
├── fake_news_eda.py
├── data/            # True.csv, Fake.csv (or link to the source)
└── outputs/         # the 5 PNG graphs
```

## Next steps
Remove the Reuters tag and `subject`, then build a TF-IDF + logistic regression classifier (needs scikit-learn, which is outside this project's library limit).
