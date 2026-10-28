## Overview

This project applies explainable data science (XDS) techniques to fake news detection. The goal is to understand *why* a model classifies an article as fake or real, rather than treating it as a black box, and to check whether those reasons hold up.

## Who Reads the Explanation and What They Decide

The explanation is read by a trust-and-safety reviewer at a news-sharing platform. For each article the screen flags as fake, the reviewer decides whether to demote it in the feed at once or to send it to manual fact-checking first. Demoting a real article suppresses a legitimate publisher, with reputational and press-freedom costs; sending every flag to fact-checking is slow and lets fabricated stories circulate in the meantime. The explanation must therefore show which words drove the flag, so that a flag hanging on one or two incidental words goes to a fact-checker rather than being acted on at once.

## Dataset

We use the ISOT Fake and Real News dataset (Ahmed, Traore & Saad, 2017, 2018), obtained from its Kaggle mirror: <https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset>. Real articles come from Reuters (`Dataset/True.csv`, 21,417 articles) and fake articles from websites flagged as unreliable (`Dataset/Fake.csv`, 23,481 articles). Each row has a title, text, subject and date. The labels describe the *source* of an article (Reuters vs. a flagged site), not a fact-check of each article. This replaces the dataset named in our proposal (see the report, Section 2). Licence: the Kaggle page should be checked before redistribution; the ISOT authors ask that the two papers above be cited.

The data is prepared in three steps:

1. `combine_dataset.ipynb` merges the two files and adds the label (`fake = 1`, `real = 0`).
2. `dataset_eda.ipynb` explores the data and looks for shortcuts that would let a model spot the label without reading the article.
3. `clean_dataset.ipynb` removes empty texts, duplicates and badly scraped rows, and strips the "CITY (Reuters) -" dateline that appears in almost every real article, leaving 38,633 articles (45.2% fake).

The EDA found that `subject` and `date` each give the label away, so they are never used as features. Other source fingerprints stay in the text on purpose, such as photo credits on fake sites ("Featured image via Getty Images") and "Reuters" in article bodies, so the explanation methods can show whether the models rely on them.

## Models

Each article's title and text are converted into TF-IDF word features (`preprocessing.py`). We train an interpretable baseline (logistic regression), a shallow decision tree that can be read in full, and four more complex models: Random Forest, LightGBM, XGBoost and an RBF-kernel SVM. All models share the same features and train/test split and are evaluated mainly on Fake-class recall, the share of fake articles caught.

**Explained model.** As set out in the proposal, the model with the highest Fake-class recall, the RBF SVM, is the one whose explanations are tested in full (`Model/XAI Evaluation/xai_evaluation.ipynb`). Its lead over the logistic regression baseline is small (0.6 recall points, 21 fewer missed fakes out of 3,489, and 5 fewer false alarms) but consistent on this split (exact McNemar test on the fake articles, p = 4.9 × 10⁻⁵).

## Explanation pipeline

Each complex model is explained with the same pipeline:

- **Global importance:** grouped Permutation Feature Importance (with a random-term check to separate real effects from noise) and SHAP (TreeSHAP for tree models, KernelSHAP for the SVM)
- **Feature effects:** ALE
- **Heterogeneity:** ICE and centered ICE
- **Individual decisions:** SP-LIME, LIME and SHAP waterfalls, compared on the same articles

A correlation check on the top terms found that several of them come as packages: the words of the photo credit (image, featured, getty, via) and of embedded tweet links (com, pic, twitter). This rules out two methods that assume the words are independent. So PFI shuffles each correlated group together instead of one term at a time, ALE replaces PDP, and ICE, which has no ALE version, only uses terms outside these groups.

The readable decision tree also serves as a test case: because its true rules are known, we can check whether the explanation methods report them correctly before trusting them on models we cannot read.

## Main findings

| Model | Fake recall | Accuracy | Missed fakes | False alarms |
|---|---|---|---|---|
| Logistic regression (baseline) | 0.969 | 0.980 | 108 | 44 |
| Decision tree (depth 3) | 0.923 | 0.883 | 269 | 639 |
| Random Forest | 0.935 | 0.966 | 228 | 33 |
| LightGBM | 0.966 | 0.980 | 119 | 35 |
| XGBoost | 0.965 | 0.980 | 122 | 33 |
| SVM (RBF, explained model) | 0.975 | 0.984 | 87 | 39 |

All models separate fake from real well, and the SVM is the most accurate, though only slightly ahead of the baseline. The explanations show what that accuracy rests on (`Model/XAI Synthesis/xai_synthesis.ipynb` compares them across models and methods):

- **Shared anchors.** All six models rely on the same four words in the same directions: "said" toward real; "via", "image" and "video" (photo credits and embedded videos) toward fake. The decision tree's very first question is whether an article contains "via".
- **Different reliance.** The tree ensembles put 33–38% of their LIME weight on source and format artefacts, the SVM 20% and logistic regression 14%; the SVM and the baseline spread 45–46% of their weight over topic and content words, and their explanations resemble each other (Spearman 0.69). LightGBM and XGBoost explain almost identically (0.95). By grouped PFI, the boosted models lose three times as much recall as the Random Forest when the photo credit is shuffled (0.061 vs 0.019), although LIME gives the photo-credit words a slightly larger share in the forest (20% vs 16–17%): the forest uses the photo credit as much but has other words to fall back on.
- **Absence counts.** 44–49% of SHAP attribution is on words an article does not contain, above all a missing photo credit and a missing "said". LIME only deletes words that are present, so it cannot show this; LIME and TreeSHAP therefore agree only partly on LightGBM (same top word 62%), while LIME and KernelSHAP, which ask the same question, agree on the SVM (same top word 100%).
- **Errors have a signature.** Correct SVM flags rest on page furniture (an artefact among the top three words for 92%); wrong flags rest on topic words and quoted speech (28%). Missed fakes are explained exactly like real Reuters articles: wire conventions carry 62% vs 61% of the supporting weight.

The models' worst mistakes show the limit. A fake site republishing a wire-style report gets through, and a real article written casually gets flagged. The models have learned to tell Reuters from these particular fake sites, which works very well within this dataset, but they judge how an article is written and laid out, not whether what it says is true. For the reviewer: a flag whose highlighted words are page furniture is usually right; a flag that rests on names, topics and quoted speech, or that disappears when two or three editorial words are deleted, should go to a fact-checker before the article is demoted.

## Evidence that the explanations can be trusted

`Model/XAI Evaluation/xai_evaluation.ipynb` tests the explanations themselves:

| Test | Result |
|---|---|
| White-box check, logistic regression | LIME vs. exact word contributions: Spearman ρ 0.81, 84% of top-5 words correct, 99% correct sign |
| White-box check, decision tree | median 99.4% of LIME's weight on the tree's 5 terms; TreeSHAP exactly 0 on the other 4,995 |
| Deletion curve, SVM (100 articles) | deleting LIME's top words moves the decision 19.5 times as far as deleting random words (AOPC 0.85 vs 0.04) |
| Seed stability, SVM (20 articles × 10 seeds) | top-5 Jaccard 0.78 at 500 LIME samples, 0.87 at 2,000; top word agrees in 91% / 94% of runs |
| Small input edits (5% of tokens) | explanation changes no more than between two seeds (Jaccard 0.877 vs 0.872) |
| Counterfactuals (editorial words only) | 64% of wrongly flagged real articles are cleared by deleting ≤ 3 words, vs 2.5% of correctly flagged fakes; as triage no better than a score threshold, but it names the words to check |
| Human forward-simulation study | **pending**: materials and protocol in `human_eval/`, no participant data yet |

## Reproducing the results

Tested with Python 3.14.2 on Windows (original runs) and Linux (re-run). Random seed: `RANDOM_STATE = 42` in `preprocessing.py`, used by every split, model and explainer; the explainer-stability tests in `xai_evaluation.ipynb` deliberately vary the LIME/KernelSHAP seeds (0 to 9) and say so.

```bash
python -m venv env && source env/bin/activate      # Windows: env\Scripts\activate
pip install -r requirements.txt
bash run_all.sh                                    # runs every notebook in order
```

`preprocessing.py` downloads the NLTK stopword and WordNet corpora on first import if they are missing. `run_all.sh` writes the report figures to `figures/` and every reported number to `results/xai_evaluation_results.json` and `results/xai_synthesis_results.json`. On a 2-core Linux machine the data, model and evaluation notebooks took 89 minutes and the synthesis notebook 12 more, mostly the SVM: it is trained in `SVM.ipynb` and again in `xai_evaluation.ipynb` (about 15 minutes each), which then caches it in `Model/XAI Evaluation/svm_rbf.joblib`. Logistic regression, the decision tree and the SVM reproduce exactly across platforms; the tree ensembles can differ by a few test articles (up to 9 false alarms for XGBoost) because of multithreaded floating-point summation and the NLTK data version (our Linux run: 198 stopwords, WordNet 3.0).

See `MODEL_CARD.md` for intended use, training data, evaluation results and known limitations.

## Project Structure

| Step | File | What it does |
|---|---|---|
| 1 | `combine_dataset.ipynb` | Merges `True.csv` and `Fake.csv` and adds the label |
| 2 | `dataset_eda.ipynb` | Explores the data and lists problems (changes nothing) |
| 3 | `clean_dataset.ipynb` | Fixes those problems and saves the cleaned dataset |
| 4 | `preprocessing.py` | Shared text cleaning, TF-IDF and train/test split |
| 5 | `Model/Baseline Model/` | Logistic regression baseline |
| 6 | `Model/Common Models/` | Decision tree, Random Forest, LightGBM, XGBoost, SVM |
| 7 | `Model/XAI Evaluation/xai_evaluation.ipynb` | Faithfulness, stability, counterfactual and ethics tests; human-study materials |
| 8 | `Model/XAI Synthesis/xai_synthesis.ipynb` | Cross-model and cross-method comparison of the explanations; explanation profiles of right and wrong predictions |
| 9 | `human_eval/` | Forward-simulation study forms, protocol and (once collected) responses |

Run steps 1–3 once, in order. The model notebooks can then be run in any order; `run_all.sh` does all of it.

## References

- Ahmed, H., Traore, I., & Saad, S. (2017). Detection of online fake news using n-gram analysis and machine learning techniques. *ISDDC 2017*, LNCS 10618, 127–138. Springer.
- Ahmed, H., Traore, I., & Saad, S. (2018). Detecting opinion spams and fake news using text classification. *Security and Privacy*, 1(1), e9.
