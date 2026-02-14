# Copilot instructions for this repository

This project is a small ML notebook-based pipeline for fraud detection using scikit-learn. Keep guidance short and immediately actionable for code-generation agents.

- **Big picture:** Data is prepared in `final_pro.ipynb` and two preprocessing artifacts are saved to disk: `type_onehoted.joblib` and `data_scaled.joblib`. Models and evaluation helpers live in `models.py`.

- **Key files:**
  - [final_pro.ipynb](final_pro.ipynb) : canonical data-prep, feature engineering, and example usage (uses `joblib.dump`).
  - [models.py](models.py) : dictionary of candidate estimators (`classifications_models`, `regression_models`) and runner helpers `run_classification_models` and `run_regression_models` (they expect X_train, X_test, y_train, y_test and return a pandas DataFrame sorted by metric).
  - `data.csv`, `type_onehoted.joblib`, `data_scaled.joblib` : primary dataset and serialized preprocessors.

- **Common patterns to follow when editing/adding code:**
  - Use the existing model dictionaries (add entries only as new keys mapping to sklearn estimators). Example: `classifications_models['MyModel'] = MyEstimator(...)`.
  - Use `joblib.load`/`joblib.dump` to read/write preprocessors and small models. Example load: `from joblib import load\nenc = load('type_onehoted.joblib')`.
  - Preserve function signatures in `models.py` (agents should call `run_classification_models(models_dict, X_train, X_test, y_train, y_test)` to evaluate).
  - Keep randomness reproducible: most estimators use `random_state=42` in this repo—follow that unless there is reason not to.

- **Dependencies & runtime:**
  - Notebook relies on: `pandas`, `numpy`, `scikit-learn`, `joblib`, `seaborn`, `matplotlib`, `imbalanced-learn` (SMOTE). Suggested install: `pip install pandas numpy scikit-learn joblib seaborn matplotlib imbalanced-learn`.
  - To reproduce data prep, open `final_pro.ipynb` in Jupyter/VS Code and run cells top-to-bottom; the notebook saves the preprocessors used later by code in this repo.

- **Dataflow / integration points:**
  - `final_pro.ipynb` reads `data.csv`, samples/preprocesses it, creates one-hot encoding for `type` and scaling for numeric columns, then saves those preprocessors as `.joblib` files.
  - Any new script or module should load the `.joblib` preprocessors and apply them consistently before passing features into models from `models.py`.

- **Testing & debugging tips specific to this repo:**
  - Re-run the notebook cell that calls `dump(...)` if the `.joblib` files are missing or outdated.
  - For quick model comparisons use the helpers in `models.py` rather than re-implementing evaluation loops.
  - When adding a model, run it locally in the notebook or a small script to confirm serialization compatibility.

- **Conventions:**
  - Do not change the names of saved artifacts (`type_onehoted.joblib`, `data_scaled.joblib`) unless also updating all callers.
  - Keep model parameter choices conservative — follow the existing hyperparameter style (explicit values, `random_state=42`).

- **Examples agents can insert:**
  - Loading preprocessors: `from joblib import load\nenc = load('type_onehoted.joblib')\nscaler = load('data_scaled.joblib')`
  - Evaluating models: `from models import classifications_models, run_classification_models\nresults = run_classification_models(classifications_models, X_train, X_test, y_train, y_test)`

If anything in these notes is unclear or you need more examples (e.g., where to place a new training script or CI commands), tell me which area to expand. 
