# Fake News Detection with Explainable AI

This project applies explainable data science (XDS) techniques to fake news detection. The goal is to understand *why* a model classifies an article as fake or real, rather than treating it as a black box, and to check whether those reasons hold up.

## Who Reads the Explanation and What They Decide

The explanation is read by a trust-and-safety reviewer at a news-sharing platform. For each article the screen flags as fake, the reviewer decides whether to demote it in the feed at once or to send it to manual fact-checking first. Demoting a real article suppresses a legitimate publisher, with reputational and press-freedom costs; sending every flag to fact-checking is slow and lets fabricated stories circulate in the meantime. The explanation must therefore show which words drove the flag, so that a flag hanging on one or two incidental words goes to a fact-checker rather than being acted on at once.

## Repository Structure

Go through the files in the numbered order: data first, then the models, then the two notebooks that compare and test the explanations.

```text
.
├── README.md                    project overview (this file)
├── MODEL_CARD.md                intended use, data, results and limitations of the explained model
├── requirements.txt             Python packages, pinned versions
├── preprocessing.py             shared code: load raw data, clean text, TF-IDF, train/test split
│
├── dataset_eda.ipynb            1. explore the raw data and list its problems (changes nothing)
├── clean_dataset.ipynb          2. combine the raw files, fix the problems, save the cleaned dataset
│
├── Dataset/
│   ├── True.csv                 raw real articles (Reuters), 21,417
│   ├── Fake.csv                 raw fake articles (flagged sites), 23,481
│   └── fake_news_dataset_cleaned.csv.gz   cleaned dataset used by every model, 38,633 articles
│
└── Model/
    ├── Baseline Model/
    │   └── Logistic_regression_TF-IDF.ipynb   3. interpretable baseline
    ├── Common Models/
    │   ├── Decision_tree.ipynb                4. depth-3 tree, readable in full (white-box test case)
    │   ├── Random_forest.ipynb                5. black-box model
    │   ├── LightGBM.ipynb                     6. black-box model
    │   ├── XGBoost.ipynb                      7. black-box model
    │   └── SVM.ipynb                          8. RBF SVM, the explained model (highest Fake recall)
    ├── Unique Models/
    │   └── EBM_CatBoost.ipynb                 9. glass-box EBM vs black-box CatBoost on stylistic features
    ├── XAI Synthesis/
    │   └── xai_synthesis.ipynb                10. what the explanations show, across models, methods and outcomes
    └── XAI Evaluation/
        └── xai_evaluation.ipynb               11. can the explanations be trusted: faithfulness, stability,
                                                   counterfactuals, ethics, human-study materials
```

Every notebook is saved with its outputs, so all results, tables and figures can be read on GitHub without running anything.

## How to Run

```bash
python -m venv env
env\Scripts\activate              # macOS/Linux: source env/bin/activate
pip install -r requirements.txt
```

Open the notebooks in VS Code (Jupyter extension) or Jupyter Notebook (`pip install notebook`) and run them in the numbered order above. A notebook can also be run from the repository root without opening it:

```bash
jupyter nbconvert --to notebook --execute --inplace "Model/Common Models/SVM.ipynb"
```

- The cleaned dataset is already in `Dataset/`, so the model notebooks (3 to 11) can be run without running 1 and 2 first.
- Notebooks 3 to 9 are independent and can be run in any order.
- `preprocessing.py` downloads the NLTK stopword and WordNet corpora on first import if they are missing.
- A full run takes about 100 minutes on a 2-core machine, mostly the SVM, which is trained in `SVM.ipynb` and again in `xai_evaluation.ipynb` (about 15 minutes each). `xai_evaluation.ipynb` then caches it as `Model/XAI Evaluation/svm_rbf.joblib` (ignored by git), and `xai_synthesis.ipynb` reuses the cache, so running notebook 11 before 10 saves about 15 minutes.

## Dataset

We use the ISOT Fake and Real News dataset (Ahmed, Traore & Saad, 2017, 2018), obtained from its Kaggle mirror: <https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset>. Real articles come from Reuters (`Dataset/True.csv`) and fake articles from websites flagged as unreliable (`Dataset/Fake.csv`). Each row has a title, text, subject and date. The labels describe the *source* of an article (Reuters vs. a flagged site), not a fact-check of each article. This replaces the dataset named in our proposal. The ISOT authors ask that the two papers above be cited.

The data is prepared in two notebooks. Both read the raw files through `load_raw_dataset()` in `preprocessing.py`, which adds the label (`fake = 1`, `real = 0`) and combines the two files in memory.

1. `dataset_eda.ipynb` explores the data and looks for shortcuts that would let a model spot the label without reading the article. It changes and saves nothing.
2. `clean_dataset.ipynb` fixes what the EDA found and saves `Dataset/fake_news_dataset_cleaned.csv.gz`: it strips extra spaces, removes the "CITY (Reuters) -" dateline that starts 99.2% of real articles, and drops rows with an invalid date (10), empty texts (631) and duplicate texts (5,624), leaving 38,633 articles (45.2% fake). The file is gzip-compressed to stay under GitHub's 100 MB limit; `pd.read_csv` reads it directly.

The EDA found that `subject` and `date` each give the label away, so they are never used as features. Other source fingerprints stay in the text on purpose, such as photo credits on fake sites ("Featured image via Getty Images") and "Reuters" in article bodies, so the explanation methods can show whether the models rely on them.

## Shared Rules for Every Model Notebook

1. **Data:** load `Dataset/fake_news_dataset_cleaned.csv.gz`; use title + text only, never `subject` or `date`.
2. **Preprocessing:** import from `preprocessing.py`; no notebook has its own text cleaning, TF-IDF settings or split.
   - `preprocess_text()`: lowercase, replace punctuation and non-ASCII characters (including curly quotes) with spaces, convert accented letters, remove numbers, tokenize, remove stopwords, lemmatize.
   - `get_split_tfidf()`: stratified 80/20 split first, then TF-IDF (unigrams, up to 5,000 terms) fitted on the training text only, so no test information leaks into the vocabulary.
3. **Seed:** always `RANDOM_STATE = 42`. The only exception is the stability test in `xai_evaluation.ipynb`, which varies the explainer seed on purpose.
4. **Metric:** recall on the Fake class (`label_encoded = 1`), always reported together with precision and accuracy, since recall alone can be maximised by flagging everything.

## Models

| # | Notebook | Why it is here | How it is explained |
|---|---|---|---|
| 3 | Logistic regression (baseline) | interpretable by design | coefficients; fingerprint check (retrain without source-fingerprint terms) |
| 4 | Decision tree, depth 3 | readable in full; its true rules are known, so it tests whether the explanation methods report them correctly | the tree itself, PFI, TreeSHAP, LIME |
| 5–8 | Random Forest, LightGBM, XGBoost, RBF SVM | more complex models that cannot be read directly | the post-hoc pipeline below |
| 9 | EBM and CatBoost (`Model/Unique Models/EBM_CatBoost.ipynb`) | the proposal's glass box vs black box, on 14 engineered stylistic features (title length, capitals, ! and ?, first/second person, word and sentence length, ...) plus the same top-30 terms | EBM shape functions (native) checked with PFI; CatBoost TreeSHAP and ALE; tabular LIME on the EBM against its exact contributions |

Models 3 to 8 share the same TF-IDF features; the EBM and CatBoost use the stylistic features plus the top-30 terms. All models use the same train/test split and are evaluated mainly on Fake-class recall, the share of fake articles caught. XGBoost is trained and explained on dense arrays (it treats sparse zeros as missing); the SVM is trained on the sparse matrix, uses `decision_function` instead of probabilities, and runs PFI and ALE on a 1,000-article sample because it is slow to score.

**Explained model.** As set out in the proposal, the model with the highest Fake-class recall, the RBF SVM, is the one whose explanations are tested in full (`xai_evaluation.ipynb`). Its lead over the logistic regression baseline is small (0.6 recall points, 21 fewer missed fakes out of 3,489, and 5 fewer false alarms) but consistent on this split (exact McNemar test on the fake articles, p = 4.9 × 10⁻⁵).

**Not built.** The Word2Vec logistic regression from the proposal; TF-IDF stays the baseline, as planned.

## Explanation Pipeline

Each complex model notebook (5 to 8) uses the same steps:

- **Top-K terms:** the 30 terms most linked to the label by chi-squared on the training data (`select_top_k_terms()`), shared by all models
- **Correlation check:** several top terms come as packages, the words of the photo credit (image, featured, getty, via) and of embedded tweet links (com, pic, twitter). This rules out two methods that assume the words are independent, so PFI shuffles each correlated group together, ALE replaces PDP, and ICE, which has no ALE version, only uses terms outside these groups.
- **Global importance:** grouped Permutation Feature Importance (with a 30-random-term check to separate real effects from noise) and SHAP (TreeSHAP for tree models, KernelSHAP over the top 30 terms for the SVM)
- **Feature effects and heterogeneity:** ALE for the top 6 terms; ICE and centered ICE
- **Individual decisions:** SP-LIME, then LIME for the most confident caught fake, the most clearly missed fake and the most confident false alarm, and a SHAP waterfall of the missed fake to compare with LIME on the same article

The two XAI notebooks then work across models:

- **`xai_synthesis.ipynb`** runs one LIME setting (1,000 samples, top 10 words) for all six models on the same 200 test articles and groups words into four categories fixed beforehand from the EDA and baseline notebooks (source artefacts, wire-service conventions, partisan/conversational style, other). It compares the models' explanations, compares LIME with SHAP, and profiles the explanations of the SVM's correct and wrong predictions.
- **`xai_evaluation.ipynb`** tests whether the explanations deserve to be believed: SVM vs baseline on the same test set, faithfulness (white-box checks, deletion curve), stability (seeds and small input edits), counterfactuals, error rates by group mention, and the materials for the human study.

## Main Findings

| Model | Fake recall | Accuracy | Missed fakes | False alarms |
|---|---|---|---|---|
| Logistic regression (baseline) | 0.969 | 0.980 | 108 | 44 |
| Decision tree (depth 3) | 0.923 | 0.883 | 269 | 639 |
| Random Forest | 0.935 | 0.966 | 228 | 33 |
| LightGBM | 0.966 | 0.980 | 119 | 35 |
| XGBoost | 0.965 | 0.980 | 122 | 33 |
| SVM (RBF, explained model) | 0.975 | 0.984 | 87 | 39 |
| EBM (stylistic features + 30 terms) | 0.986 | 0.991 | 48 | 19 |
| CatBoost (stylistic features + 30 terms) | 0.989 | 0.993 | 40 | 15 |

All models separate fake from real well. Among the TF-IDF models the SVM is the most accurate, though only slightly ahead of the baseline; the two models on stylistic features are more accurate still. The explanations show what that accuracy rests on (`xai_synthesis.ipynb`, `EBM_CatBoost.ipynb`):

- **Shared anchors.** All six models rely on the same four words in the same directions: "said" toward real; "via", "image" and "video" (photo credits and embedded videos) toward fake. The decision tree's very first question is whether an article contains "via".
- **Different reliance.** The tree-based models put 33–38% of their LIME weight on source and format artefacts, the SVM 20% and logistic regression 14%; the SVM and the baseline spread 45–46% of their weight over topic and content words, and their explanations resemble each other (Spearman 0.69). LightGBM and XGBoost explain almost identically (0.95). By grouped PFI, the boosted models lose three times as much recall as the Random Forest when the photo credit is shuffled (0.061 vs 0.019), although LIME gives the photo-credit words a slightly larger share in the forest (20% vs 16–17%): the forest uses the photo credit as much but has other words to fall back on.
- **Absence counts.** 44–49% of SHAP attribution is on words an article does not contain, above all a missing photo credit and a missing "said". LIME only deletes words that are present, so it cannot show this; LIME and TreeSHAP therefore agree only partly on LightGBM (same top word 62%), while LIME and KernelSHAP, which ask the same question, agree on the SVM (same top word 100%).
- **Errors have a signature.** Correct SVM flags rest on page furniture (an artefact among the top three words for 92%); wrong flags rest on topic words and quoted speech (28%). Missed fakes are explained exactly like real Reuters articles: wire conventions carry 62% vs 61% of the supporting weight.
- **Style: glass box and black box agree.** On the stylistic features, the glass-box EBM is within 0.2 recall points of CatBoost, so the gain over the TF-IDF models comes with the features (capitals, punctuation, length that TF-IDF preprocessing removes), not with model complexity. Both models rely most on the title, its share of capitalised words and its length, ahead of "said" and the photo credit. Stylistic features take 41% (EBM) and 46% (CatBoost) of the importance, the two importance rankings correlate at 0.90, and CatBoost's ALE curves follow the EBM's native shape functions (r 0.92–0.95). A quotation-mark feature was left out because ISOT's fake bodies lost their quotation marks in scraping.

The models' worst mistakes show the limit. A fake site republishing a wire-style report gets through, and a real article written casually gets flagged. The models have learned to tell Reuters from these particular fake sites, which works very well within this dataset, but they judge how an article is written and laid out, not whether what it says is true. For the reviewer: a flag whose highlighted words are page furniture is usually right; a flag that rests on names, topics and quoted speech, or that disappears when two or three editorial words are deleted, should go to a fact-checker before the article is demoted.

## Evidence that the Explanations Can Be Trusted

From `xai_evaluation.ipynb` (the EBM row from `EBM_CatBoost.ipynb`):

| Test | Result |
|---|---|
| White-box check, logistic regression | LIME vs. exact word contributions: Spearman ρ 0.81, 84% of top-5 words correct, 99% correct sign |
| White-box check, decision tree | median 99.4% of LIME's weight on the tree's 5 terms; TreeSHAP exactly 0 on the other 4,995 |
| White-box check, EBM (50 articles) | tabular LIME vs the EBM's exact contributions: ρ 0.82 and the correct sign for every top-5 feature, but only 35% of the exact top 5 |
| Deletion curve, SVM (100 articles) | deleting LIME's top words moves the decision 19.5 times as far as deleting random words (AOPC 0.85 vs 0.04) |
| Seed stability, SVM (20 articles × 10 seeds) | top-5 Jaccard 0.78 at 500 LIME samples, 0.87 at 2,000; top word agrees in 91% / 94% of runs |
| Small input edits (5% of tokens) | explanation changes no more than between two seeds (Jaccard 0.877 vs 0.872) |
| Counterfactuals (editorial words only) | 64% of wrongly flagged real articles are cleared by deleting ≤ 3 words, vs 2.5% of correctly flagged fakes; as triage no better than a score threshold, but it names the words to check |
| Human forward-simulation study | **pending**: Section 7 of the notebook selects the 10 articles and writes the participant forms when `MAKE_STUDY_FORMS = True`; no participant data yet |

## Human Study (To Do)

**Status: not run yet.** Everything else in the project is done; this is the one piece of evidence still missing. Rough plan, as in the proposal:

1. **Make the forms.** In `Model/XAI Evaluation/xai_evaluation.ipynb`, Section 7, set `MAKE_STUDY_FORMS = True` and run the notebook. It writes `human_eval/form_A.html`, `form_B.html`, `answer_key.csv` (for the organiser only, never shown to participants) and an empty `response_template.csv`.
2. **Recruit 10 people.** Classmates are fine; people with newsroom, moderation or fact-checking experience are better. Give P01, P03, ... Form A and P02, P04, ... Form B, so 5 people use each form. Ask for verbal consent and record only an ID and a one-word background.
3. **Run each session (about 20 minutes).** Tell the participant to predict what the *screen* will say, not whether the article is true. They read two practice items, then for each of the 10 items tick FAKE or REAL and give a confidence from 1 to 5. Note the seconds per item and anything they say about the highlighted words. Give no feedback until the end.
4. **Record.** Save `human_eval/responses.csv`, one row per participant and item, with the template's columns: `participant_id, form, background, item, predicted_model_output, confidence_1to5, seconds, comment`.
5. **Score and report.** Re-run Section 7.1 of the notebook. It reports how often participants predicted the screen correctly with and without the explanation, overall and on the screen's errors. Put those numbers, the sample size and two or three participant comments into the report (Section 5, Table 2) and `MODEL_CARD.md`. Do not claim statistical significance with 10 people.

Each form shows 5 of the 10 articles with the LIME words highlighted and 5 without, so every person sees both conditions. The 10 articles were chosen on purpose (3 correct flags, 2 false alarms, 3 correctly cleared, 2 missed fakes), so judging the truth alone does not score well.

## Reproducibility

Tested with Python 3.14.2 on Windows (original runs) and Linux (re-run). Random seed: `RANDOM_STATE = 42` in `preprocessing.py`, used by every split, model and explainer. Logistic regression, the decision tree and the SVM reproduce exactly across platforms; the tree ensembles can differ by a few test articles (up to 9 false alarms for XGBoost) because of multithreaded floating-point summation and the NLTK data version (our Linux run: 198 stopwords, WordNet 3.0).

## References

- Ahmed, H., Traore, I., & Saad, S. (2017). Detection of online fake news using n-gram analysis and machine learning techniques. *ISDDC 2017*, LNCS 10618, 127–138. Springer.
- Ahmed, H., Traore, I., & Saad, S. (2018). Detecting opinion spams and fake news using text classification. *Security and Privacy*, 1(1), e9.
