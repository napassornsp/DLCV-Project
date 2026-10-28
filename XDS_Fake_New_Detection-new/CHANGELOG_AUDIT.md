# Audit Change Log (2026-10-28)

Final quality-assurance pass on commit `f255154`. Originals are preserved in git history; every change below is limited to the cell or file named. No reported number in an existing notebook was changed.

## Code

| File | Cell | Change | Reason |
|---|---|---|---|
| `preprocessing.py` | – | `_ensure_nltk_resources()` downloads the stopword and WordNet corpora on import if missing | A fresh clone failed in `combine_dataset.ipynb` with `LookupError` (verified in a clean environment) |
| `requirements.txt` | – | added `nbclient==0.10.2`, `nbconvert==7.17.2`, `nbformat==5.11.1` | needed by `run_all.sh` |
| `run_all.sh` | new | runs every notebook in order with `nbconvert --execute --inplace` | one-command reproduction (criterion 9) |
| `.gitignore` | – | ignores `Model/XAI Evaluation/svm_rbf.joblib` | cached model written by the evaluation notebook |
| `Random_forest.ipynb`, `LightGBM.ipynb`, `XGBoost.ipynb`, `SVM.ipynb` | `e7c8054d`, `24c4a6fc`, `3ad0789c`, `bc834dd4` | removed the hard-coded print label `<- large gap vs test = overfitting`; the same label text was removed from the stored output, all numbers untouched | the label contradicted the Markdown ("the gap is small"); re-run to regenerate the output |
| `Model/XAI Evaluation/xai_evaluation.ipynb` | new | model-vs-baseline test, white-box and deletion faithfulness tests, LIME/KernelSHAP stability, input perturbation, counterfactuals, group-mention error rates, human-study materials | criteria 2–8 had no quantitative evidence about the explanations |
| `figures/`, `results/`, `human_eval/` | new | outputs of the evaluation notebook; `human_eval/PROTOCOL.md` | report figure and numbers must be regenerable |

## Markdown

| File | Cell | Change |
|---|---|---|
| `Logistic_regression_TF-IDF.ipynb` | `11f91bad` | deleted: misplaced duplicate paragraph ("The check below answers that" after the check) |
| `Logistic_regression_TF-IDF.ipynb` | `23d26c5b` | explains the baseline's role as white-box reference for the faithfulness test |
| `Decision_tree.ipynb` | `b1a4bb9f`, `83ae2508` | "we can trust them more" replaced by measured white-box results and a caveat |
| `XGBoost.ipynb` | `24db64dc` | agreement between libraries no longer presented as faithfulness evidence |
| `SVM.ipynb` | `4fad282e`, `9dccbde8` | states plainly that the SVM leads the baseline by 0.6 recall points, with McNemar/bootstrap evidence and the reason for keeping it |
| `SVM.ipynb` | `fbbafedb` | "LIME results change a little" replaced by measured stability |
| `README.md` | – | target-audience paragraph, dataset provenance and citation, explained model, evidence table, reproduction instructions, references |
| `modeling_pipeline.md` | – | EBM/CatBoost and Word2Vec marked as descoped; XAI evaluation step and NLTK rule added |
| `MODEL_CARD.md` | new | intended use, data, results, limitations (criterion 9) |

## XAI refocus (second pass, 2026-10-28)

Goal: make the explanations and their interpretation the centre of the project. No model, feature or quality test was added; the new notebook reuses the trained models, the LIME settings of the model notebooks and the cached SVM.

| File | Cell | Change | Reason |
|---|---|---|---|
| `Model/XAI Synthesis/xai_synthesis.ipynb` | new | LIME for all six models on one 200-article sample; word categories fixed in advance from the EDA and baseline notebooks; category shares, top-5 concentration, photo-credit share, Spearman similarity between models; SHAP absence share; LIME vs KernelSHAP (SVM) and LIME vs TreeSHAP (LightGBM); explanation profiles of all SVM false positives and false negatives against 100 random true positives and true negatives; interpretation after every result | the project explained each model separately but never compared explanations across models, methods or outcomes |
| `results/xai_synthesis_results.json`, `figures/fig_xai_synthesis.png` | new | outputs of the synthesis notebook (two runs, identical numbers) | report Figure 1 and every Section 4 number |
| `SVM.ipynb` | `fbbafedb` | appended one paragraph: the three LIME cases are the most extreme, and the pattern across all 87 missed fakes and 39 false alarms | the cases illustrate; the profile summarises |
| `LightGBM.ipynb` | `213fec24` | appended one paragraph: 46% of LightGBM's SHAP attribution is on absent words across 200 articles | the waterfall's "missing photo credit" finding came from one article |
| `README.md` | Main findings | four findings after the results table; findings now come before the evidence table; Project Structure row 8 | README led with performance and quality tests |
| `run_all.sh` | last lines | runs the synthesis notebook after the evaluation notebook | reproducibility |
| `modeling_pipeline.md` | new section | "XAI Synthesis" | documents the new step |

## Not changed (recommended to the team)

- `requirements-word2vec.txt` is orphaned (its notebook was removed in `5d202db`): delete it.
- Before submission, run `bash run_all.sh` once on the submitting machine so every notebook has sequential execution counts.
- Confirm the dataset licence on the Kaggle page and the reason for the dataset change (marked `[team: …]` in the report).
- Run the human study (`human_eval/PROTOCOL.md`) and add `human_eval/responses.csv`.
