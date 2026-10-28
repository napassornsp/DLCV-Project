# Modeling Pipeline Overview

This document explains the overall modeling flow for the Fake News Detection project, so every notebook is built consistently.

## Data Preparation (run once, in this order)

1. **`combine_dataset.ipynb`:** merges `Dataset/True.csv` (real, Reuters) and `Dataset/Fake.csv` (fake) into `Dataset/fake_news_combined.csv.gz`, adding `label` and `label_encoded` (`fake = 1`, `real = 0`).
2. **`dataset_eda.ipynb`:** explores the combined data and reports problems. It changes nothing and saves nothing.
3. **`clean_dataset.ipynb`:** fixes what the EDA found and saves `Dataset/fake_news_dataset_cleaned.csv.gz`:
   - strips extra spaces
   - removes the "CITY (Reuters) -" dateline from real articles
   - drops rows with an invalid date, empty texts and duplicate texts

Files are gzip-compressed to stay under GitHub's 100 MB limit; `pd.read_csv` reads them directly.

## Shared Rules (apply to every model notebook)

1. **Dataset:** always load `Dataset/fake_news_dataset_cleaned.csv.gz`.
2. **Features:** title + text only. `subject` and `date` each give the label away (see the EDA), so they are never used as features.
3. **Shared preprocessing file:** always import from `preprocessing.py`. Do not write your own text cleaning, TF-IDF settings or train/test split.
```python
   from preprocessing import (
       preprocess_text, get_split_tfidf,
       select_top_k_terms, RANDOM_STATE
   )
```
4. **Text preprocessing:** `preprocess_text()`: lowercase → replace all punctuation and non-ASCII characters (including curly quotes) with spaces → convert accented letters → remove numbers → tokenize → remove stopwords → lemmatize.
5. **TF-IDF and split:** `get_split_tfidf()`: stratified 80/20 split first, then TF-IDF (unigrams, up to 5,000 terms) fitted on the training text only, so no test information leaks into the vocabulary.
6. **Random seed:** always use `RANDOM_STATE` (currently `42`). Never hardcode a different value. (Exception: the stability tests in `xai_evaluation.ipynb` vary the explainer seed on purpose.)
7. **Primary metric:** recall on the **Fake** class (`label_encoded = 1`), reported with `recall_score(y_test, y_pred, pos_label=1)`. Always report it with precision and accuracy, since recall alone can be maxed out by flagging everything.
8. **NLTK data:** `preprocessing.py` downloads the stopword and WordNet corpora on import if they are missing, so no notebook needs its own `nltk.download()`.

---

## Baseline Model (Logistic Regression)

**Notebook:** `Model/Baseline Model/Logistic_regression_TF-IDF.ipynb`

**Why:** logistic regression is interpretable by design. Its coefficients explain predictions directly, so no post-hoc methods are needed.

**Steps:**
- Load the cleaned data, preprocess, TF-IDF and split (shared functions)
- Train logistic regression
- Evaluate (classification report, confusion matrix, Fake-class recall)
- Sanity check: dummy classifier and permutation test (shuffled labels)
- Interpret coefficients (top terms pushing toward fake and real)
- Fingerprint check: retrain without source-fingerprint terms to see how much accuracy depends on them

---

## Readable Model (Decision Tree)

**Notebook:** `Model/Common Models/Decision_tree.ipynb`

**Why:** a depth-3 tree can be read and checked in full. Because its true rules are known, it is also used to test whether the explanation methods report them correctly.

**Steps:** train (depth 3, at least 50 articles per leaf) → read the tree → compare depths with cross-validation → evaluate → PFI and TreeSHAP → LIME and SHAP waterfall on single articles. PDP, ICE and SP-LIME are skipped, because the tree itself already shows what they would.

---

## Black-Box Models (Random Forest, LightGBM, XGBoost, SVM)

**Notebooks:** `Random_forest.ipynb`, `LightGBM.ipynb`, `XGBoost.ipynb` and `SVM.ipynb`, all in `Model/Common Models/`

**Why:** these models can't be read directly, so they are explained with post-hoc methods.

**Model-specific notes:**
- **XGBoost** treats sparse zeros as missing values, so it is trained and explained on dense arrays.
- **SVM** is trained on the sparse matrix (much faster with 5,000 terms). It uses `decision_function` instead of probabilities, and PFI and ALE run on a 1,000-article sample because it is slow to score.

**Steps (same for every model):**
- Load, preprocess, TF-IDF and split (shared functions)
- Train the model
- Evaluate (classification report, confusion matrix, Fake-class recall, ROC AUC)
- **Top-K terms:** the 30 terms most linked to the label by chi-squared on the training data (`select_top_k_terms()`), shared across all models
- **Correlation check:** heatmap of the top-K terms, flagging pairs with |correlation| > 0.5. Several terms come as packages (photo-credit words, tweet links), which rules out methods that assume independent words.
- **Explanation methods:**
  - **Grouped PFI:** drop in Fake-class recall when terms are shuffled; correlated groups are shuffled together, other terms one at a time. A null check with 30 random terms separates real effects from noise.
  - **SHAP:** TreeSHAP for tree models; KernelSHAP for the SVM (top 30 terms only)
  - **ALE:** feature effects for the top 6 terms (used instead of PDP because of the correlation)
  - **ICE and centered ICE:** only for top terms outside the correlated groups, since ICE has no ALE version
  - **SP-LIME:** representative articles, chosen automatically
  - **LIME:** the most confident caught fake, the most clearly missed fake and the most confident false alarm
  - **SHAP waterfall:** the missed fake article, to compare with LIME on the same case
- Model summary

---

## Tier 3: Unique Models (EBM, CatBoost): descoped

The proposal planned an Explainable Boosting Machine and CatBoost on engineered stylistic features. They were not built: the grading criteria ask for evidence about the explanations of one explained model, and that time went into the faithfulness, stability and actionability tests below. They remain future work. The Word2Vec logistic regression from the proposal was dropped for the same reason (its notebook was removed in commit `5d202db`).

---

## XAI Synthesis (what the explanations show, across models and methods)

**Notebook:** `Model/XAI Synthesis/xai_synthesis.ipynb`

**Why:** each model notebook explains one model; this notebook compares the explanations. It uses the same models and one LIME setting (1,000 samples, top 10 words) for all six models on the same 200 test articles, and groups words into four categories fixed beforehand from the EDA and baseline notebooks (source artefacts, wire-service conventions, partisan/conversational style, other).

**Steps:**
- Across models: top words, category shares of LIME weight, concentration, explanation similarity (Spearman)
- Across methods: share of SHAP attribution on absent words; LIME vs KernelSHAP (SVM) and LIME vs TreeSHAP (LightGBM) agreement
- Across outcomes: LIME profiles of the SVM's true/false positives and negatives (all errors, 100 of each correct outcome)

Figure: `figures/fig_xai_synthesis.png`; numbers: `results/xai_synthesis_results.json`.

---

## XAI Evaluation (evidence about the explanations)

**Notebook:** `Model/XAI Evaluation/xai_evaluation.ipynb`

**Why:** the model notebooks show what the explanations say; this notebook tests whether they deserve to be believed. The explained model is the RBF SVM, chosen by the proposal's rule (highest Fake-class recall).

**Steps:**
- Retrain LR, decision tree and SVM with the same settings (the SVM is cached in `svm_rbf.joblib` after the first run)
- SVM vs baseline on the same test set: McNemar exact test, bootstrap CI, near-duplicate sensitivity
- Faithfulness: white-box checks on LR (LIME, KernelSHAP vs exact contributions) and the decision tree; deletion curve on the SVM (LIME top-k vs random words)
- Stability: LIME across 10 seeds (500 and 2,000 samples), KernelSHAP across 5 seeds, small input perturbations
- Actionability: minimal word-deletion counterfactuals for flagged articles, editorial words only vs any word
- Ethics: error rates for articles that mention protected groups; correlation of top terms with those mentions
- Human evidence: forward-simulation forms, protocol and scoring code (`human_eval/`); no participant data is generated

Figures go to `figures/`, numbers to `results/xai_evaluation_results.json`.

---

## Final Comparison

All models are compared in one table: Fake-class recall, accuracy, missed fakes, false alarms and the interpretability approach used. The final discussion covers the accuracy–interpretability trade-off and, using the explanations, what the models actually rely on: writing style versus source fingerprints.
