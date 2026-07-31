# Freight Rate ML Assessment

A machine learning pipeline that predicts freight (posted) load rates from
load, route, and market features. Built for the Spotter Machine Learning
Engineer assessment.

## Project Structure

```
.
├── freight_rate_ml.py                     # Full pipeline: EDA -> features -> training -> predictions -> scoring
├── Freight_Rate_ML.ipynb                  # Same pipeline as a notebook (Colab export)
├── data/
│   ├── train-test.csv                     # 48,000 labeled loads (development data)
│   ├── validation.csv                     # 12,000 unlabeled loads requiring final predictions
│   └── december-chart-inputs.csv          # 31-day fixed-lane template for the December chart
├── models/
│   ├── best_model.pkl                     # Trained, tuned Ridge Regression model
│   ├── scaler.pkl                         # StandardScaler fit on training features
│   └── all_models_results.pkl             # CV results for every model that was trained
├── outputs/
│   ├── exploration_analysis.png           # EDA plots (target distribution, feature relationships)
│   ├── model_comparison.csv / .png        # CV R² / MAE / RMSE comparison across models
│   ├── predicted_vs_actual.png            # Residual diagnostic: predicted vs. actual rate
│   ├── residual_distribution.png          # Residual diagnostic: error distribution
│   ├── candidate_december.png             # Fixed December-2025 rate chart (from score.py)
│   ├── validation_predictions.csv         # Final required deliverable: 12,000 predictions
│   └── december-chart-inputs.csv          # December template filled with predicted_rate
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+ is recommended. `xgboost` and `lightgbm` are optional — the
pipeline is written to skip them gracefully if not installed, but they are
included here because they were used during model comparison.

## How to Run

The pipeline was developed in Google Colab (`Freight_Rate_ML.ipynb`), so
`freight_rate_ml.py` is the Colab export with `drive.mount(...)` and
`/content/drive/...` paths. To run it locally:

1. Open `freight_rate_ml.py` (or the notebook).
2. Replace the Colab paths with the local `data/` paths, e.g.:
   ```python
   train_df = pd.read_csv('data/train-test.csv')
   val_df   = pd.read_csv('data/validation.csv')
   ...
   december_df = pd.read_csv('data/december-chart-inputs.csv')
   ```
3. Remove the `google.colab` import/mount lines at the top.
4. Run top to bottom:
   ```bash
   python freight_rate_ml.py
   ```

Running it end to end will:
- Explore and clean `train-test.csv`
- Engineer temporal, geographic, interaction, and location-aggregate features
- Train and cross-validate Ridge, Lasso, Random Forest, Gradient Boosting,
  XGBoost, and LightGBM, then tune the winning model
- Score `validation.csv` and write `validation_predictions.csv`
- Score `december-chart-inputs.csv` and write the filled version
- Validate both output files and render the fixed December chart via the
  embedded `score.py`-equivalent logic (see `main()` at the bottom of the
  script)

## Model

**Best model: Ridge Regression (tuned via `RandomizedSearchCV`)**

| Metric | Value |
|---|---|
| Cross-validation R² (5-fold) | ~0.851 |
| Training MAE | ~$150.42 |
| Training RMSE | ~$572.77 |

Ridge was selected over Random Forest, Gradient Boosting, XGBoost, and
LightGBM because it matched or slightly beat their cross-validation R²
while being far cheaper to train and fully interpretable via its
coefficients. Full comparison: `outputs/model_comparison.csv`.

## Validation Approach

- **Development data (`train-test.csv`, 48,000 rows):** split internally
  using 5-fold `KFold` cross-validation (`shuffle=True, random_state=42`)
  rather than a single train/test hold-out, so every model's reported R²
  is an average over 5 folds rather than one lucky/unlucky split.
- **Final predictions (`validation.csv`, 12,000 rows):** never used in
  training or model selection — scored only once, by the final tuned model,
  to produce `validation_predictions.csv`.
- See the report (`report.docx`) for the full write-up, including data
  quality issues found and how they were handled.

## Output Format

`validation_predictions.csv` contains exactly:

```
load_id,predicted_rate
```

for all 12,000 `load_id` values in `data/validation.csv`, with the
`predicted_rate` column filled in from the template.
