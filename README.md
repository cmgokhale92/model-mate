# ModelMate — AutoML MVP

Business users upload a CSV/Excel file, pick the column to predict, press **TRAIN MODEL**, and later
score new data from the **Models** screen. Brief: [planning.md](planning.md).

## Run

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m uvicorn backend.main:app --reload
```
Open http://127.0.0.1:8000 (API docs at `/docs`).

## Try it with the sample data
1. **Data uploader**: drop `sample_data/insurance_churn_train.csv` (3,000 insurance customers, ~27% churn).
   Browse the four tabs: Data (editable), File details, Statistics, Charts.
2. Enter a name and description, choose **churn** as the target, then press **TRAIN MODEL** (~15 s).
3. **Models**: click the card, then upload `sample_data/insurance_churn_new_customers.csv` (300 unseen customers).
   A predictions file downloads with `predicted_churn`, `probability_churn_Yes` and `priority_rank`
   (1 = contact first). Compare against `insurance_churn_new_customers_truth.csv` if you like.

Regenerate the data with `python sample_data/generate_sample_data.py`.

## Tests
```powershell
.venv\Scripts\python -m pytest -v
```
`tests/test_modelmate.py` covers the full flow: upload, stats and charts, cell edits, training, the model registry,
the template, predictions on unseen data (AUC > 0.70 against the true labels), schema-mismatch errors and delete.
It also checks task detection and a regression target. Tests use a temporary data directory.

## How BACKEND JOB 1 works (`backend/automl.py`)
| Step | What happens |
|---|---|
| Clean | Drops rows with no target and duplicate rows. Drops ID-like, constant and >60%-missing columns. |
| Modalities | Detects numeric, categorical, boolean, datetime (split into year, month, weekday and age) and free-text (TF-IDF) columns. Imputation and scaling or one-hot encoding happen inside the pipeline. |
| Target | Detects binary, multiclass or regression. Picks the positive class (Yes/1/True, else the minority class). Uses balanced class weights when classes are imbalanced. |
| CV strategy | Stratified K-fold for classification, K-fold for regression. Uses 3 folds instead of 5 above 20k rows. Above 30k rows, model selection runs on a sample. |
| Candidates | Baseline, Logistic/Ridge, Random Forest, Extra Trees, Histogram Gradient Boosting. |
| Selection | Primary metric is ROC AUC (binary), macro F1 (multiclass) or R² (regression). The best model gets a randomized-search tuning pass and keeps the tuned settings only if they score higher. |
| Output | Refits on all rows, computes permutation feature importance and saves the full pipeline with joblib to `data/models/<id>.joblib`. The SQLite row in `data/modelmate.db` stores the name, description, schema, metrics, leaderboard and decisions. |

## Layout
```
backend/   main.py (API + serves frontend), automl.py, profiling.py, db.py, config.py
frontend/  index.html, styles.css, app.js   (no build step, no CDN)
sample_data/  generator + train / new-customer / truth CSVs
tests/     test_modelmate.py
data/      created at runtime: uploads, model artifacts, SQLite DB
```

## MVP limitations
- Training jobs are tracked in memory, so restarting the server loses the status of a running job. Saved models persist.
- There is no authentication or per-user model ownership yet.
