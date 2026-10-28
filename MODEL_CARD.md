# Model Card: RBF-SVM Fake-News Screen (TF-IDF)

Format follows Mitchell et al. (2019), "Model Cards for Model Reporting". All numbers come from `Model/Common Models/SVM.ipynb` and `Model/XAI Evaluation/xai_evaluation.ipynb`.

## Model details

- **Type:** support vector machine, RBF kernel, `C = 1.0`, `gamma = "scale"` (1.014 on this data), scikit-learn 1.9.1.
- **Input:** title + text, lower-cased, punctuation, digits and non-ASCII removed, stopwords removed, lemmatised (`preprocessing.py`); TF-IDF, unigrams, 5,000 terms, `min_df = 5`, `max_df = 0.9`, fitted on training articles only.
- **Output:** decision value (> 0 = fake). A sigmoid of it is shown as a score; it is **not** a calibrated probability.
- **Size:** 7,866 support vectors (25% of training articles) in 5,000 dimensions (7,865 in the original Windows run).
- **Seed:** `RANDOM_STATE = 42` for the split; the SVM itself is deterministic.
- **Developers:** Insight Lab, XDS 2026, Asian Institute of Technology.

## Intended use

- **Intended user:** a trust-and-safety reviewer who decides whether a flagged article is demoted at once or sent to manual fact-checking first.
- **Intended use:** a triage screen that ranks articles for human review, shown together with its LIME explanation.
- **Out of scope:** automatic removal or demotion without human review; judging whether a claim is true; any outlet other than Reuters and the fake sites in ISOT; non-English text; news after 2018.

## Training and evaluation data

- ISOT Fake and Real News dataset (Ahmed et al., 2017, 2018), Kaggle mirror by C. Bisaillon. 44,898 raw articles; 38,633 after cleaning (45.2% fake). 2015–2018, mostly US politics and world news.
- Stratified 80/20 split: 30,906 training, 7,727 test articles.
- Real = Reuters, fake = sites flagged as unreliable. **Labels describe the source, not a fact-check of each article.**

## Evaluation results (test set, 7,727 articles)

| Model | Fake recall | Precision | Accuracy | ROC AUC | Missed fakes | False alarms |
|---|---|---|---|---|---|---|
| SVM (this model) | 0.9751 | 0.989 | 0.9837 | 0.998 | 87 | 39 |
| Logistic regression baseline | 0.9690 | 0.987 | 0.9803 | 0.998 | 108 | 44 |

SVM vs baseline: +0.6 recall points (bootstrap 95% CI 0.3 to 0.9) and 5 fewer false alarms; on fake articles the SVM wins 24 of 27 disagreements (exact McNemar p = 4.9 × 10⁻⁵, one split).

A glass-box alternative exists: an EBM on 14 stylistic features plus the top-30 terms (`Model/Unique Models/EBM_CatBoost.ipynb`) reaches Fake recall 0.986 (48 missed fakes, 19 false alarms) on the same test set and can be read in full.

Explanation quality (LIME, SVM): deletion AOPC 0.85 vs 0.04 for random words; top-5 seed agreement (Jaccard) 0.78 at 500 samples, 0.87 at 2,000; on average no more change under 5% token deletion than between seeds. Human forward-simulation study: **not yet run**.

## Known limitations and risks

- **Source shortcut.** The model separates Reuters house style ("said", weekdays, titles) and fake-site page furniture (photo credits, "READ MORE", tweet links) from each other. It does not check facts. A fake site copying a wire report is missed; a casual Reuters feature is flagged.
- **Fragile flags.** 64% of wrongly flagged real articles are cleared by deleting three or fewer editorial words (2.5% of correct flags); such flags should go to a fact-checker before demotion. A decision-value threshold of 0.29 separates them equally well.
- **Evasion.** 96% of the small edits that clear a correctly flagged fake involve page-furniture words; a fake site can evade the screen by removing them.
- **Topic and period cues.** Event names (e.g. a perpetrator's surname) can decide a flag; the data covers 2015–2018 only.
- **Groups and politics.** Real articles with race-related terms were flagged at 3.6% (4 of 110) vs 0.85% otherwise; partisan naming (hillary, gop) is a top feature. Counts are small; no significance claimed.
- **Explanation stability.** the top word agrees across seeds in 91% of runs (worst article 29%); use ≥ 2,000 samples and treat words beyond the first as indicative.
- **Reproducibility.** Predictions reproduce exactly across platforms (support-vector count 7,865 vs 7,866); NLTK data version (198 stopwords, WordNet 3.0) should be kept fixed.
