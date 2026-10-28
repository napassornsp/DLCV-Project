#!/usr/bin/env bash
# Reproduce every result and figure from the raw CSV files.
#   python -m venv env && source env/bin/activate && pip install -r requirements.txt
#   bash run_all.sh
# Figures are written to figures/, all reported numbers to results/xai_evaluation_results.json and results/xai_synthesis_results.json.
set -euo pipefail
cd "$(dirname "$0")"

run() {
  echo ">> $1"
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "$1"
}

run combine_dataset.ipynb
run dataset_eda.ipynb
run clean_dataset.ipynb
run "Model/Baseline Model/Logistic_regression_TF-IDF.ipynb"
for nb in Decision_tree Random_forest LightGBM XGBoost SVM; do
  run "Model/Common Models/${nb}.ipynb"
done
run "Model/XAI Evaluation/xai_evaluation.ipynb"
run "Model/XAI Synthesis/xai_synthesis.ipynb"      # reuses the SVM cached by the evaluation notebook

echo "Done. Figures: figures/   Numbers: results/xai_evaluation_results.json, results/xai_synthesis_results.json"
