"""
Shared preprocessing pipeline for Fake News Detection project.
All model notebooks (baseline + Tier 2) must import from here to ensure
identical preprocessing, TF-IDF settings, and train/test split across models.

Row-level cleaning (duplicates, empty texts, Reuters dateline) is done once in
clean_dataset.ipynb. This file only processes the text of the cleaned dataset.
"""

import re
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_selection import chi2, mutual_info_classif

# Fixed seed used across the ENTIRE project — never change this
RANDOM_STATE = 42


def _ensure_nltk_resources():
    """Download the NLTK corpora this module needs if they are missing.

    The stopword list is loaded when this module is imported, so the corpora must
    exist before any notebook runs `from preprocessing import ...`. Without this
    check, a fresh clone fails in combine_dataset.ipynb with a LookupError.
    """
    for resource in ("stopwords", "wordnet"):
        found = False
        for path in (f"corpora/{resource}", f"corpora/{resource}.zip"):   # wordnet stays zipped
            try:
                nltk.data.find(path)
                found = True
                break
            except LookupError:
                pass
        if not found:
            nltk.download(resource, quiet=True)


_ensure_nltk_resources()

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    """Lowercase, remove punctuation/numbers/non-ASCII characters, tokenize,
    remove stopwords, lemmatize.

    Real (Reuters) articles use curly quotes, dashes, non-breaking spaces and
    accented letters, while fake articles contain no non-ASCII characters at all.
    Every non-letter character is therefore replaced with a space, so these
    characters cannot become tokens that only appear in one class.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)            # all punctuation, incl. ’ “ ” – — → space
    text = (unicodedata.normalize("NFKD", text)     # accented letters → plain letters (josé → jose)
            .encode("ascii", "ignore").decode("ascii"))
    text = re.sub(r"[^a-z\s]", " ", text)           # numbers, underscores, anything left → space

    tokens = text.split()                           # only letters and spaces remain, so split() is enough
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words]

    return " ".join(tokens)


def get_tfidf_features(df, max_features=5000, min_df=5, max_df=0.9, ngram_range=(1, 1)):
    """Fit TF-IDF on processed text. Returns X, y, and the fitted vectorizer."""
    tfidf = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df
    )
    X = tfidf.fit_transform(df["processed_text"])
    y = df["label_encoded"]
    return X, y, tfidf


def get_train_test_split(X, y, test_size=0.2):
    """Standard split used by every model in this project."""
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y
    )


def select_top_k_terms(X_train, y_train, feature_names, k=30, method="chi2"):
    """Rank TF-IDF terms by association with the label, using TRAINING data only.
    Returns (column indices, term names, scores). Shared by all Tier 2 notebooks."""
    if method == "chi2":
        scores, _ = chi2(X_train, y_train)
    elif method == "mutual_info":
        scores = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE)
    else:
        raise ValueError("method must be 'chi2' or 'mutual_info'")
    scores = np.nan_to_num(scores)
    top_idx = np.argsort(scores)[::-1][:k]
    return top_idx, np.asarray(feature_names)[top_idx], scores[top_idx]


def get_split_tfidf(df, test_size=0.2, max_features=5000, min_df=5,
                    max_df=0.9, ngram_range=(1, 1)):
    """Split first, then fit TF-IDF on training text only (no leakage).
    Returns X_train, X_test, y_train, y_test, and the fitted vectorizer."""
    text_train, text_test, y_train, y_test = train_test_split(
        df["processed_text"], df["label_encoded"],
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df["label_encoded"]
    )

    tfidf = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df
    )
    X_train = tfidf.fit_transform(text_train)   # learn vocabulary from train only
    X_test = tfidf.transform(text_test)         # apply it to test

    return X_train, X_test, y_train, y_test, tfidf