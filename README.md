# 🏠 Ames House Price Predictor

An end-to-end machine learning project that predicts residential sale prices using the [Ames Housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques), deployed as an interactive Streamlit app with SHAP-based explainability.

**🔗 Live demo: [ames-house-price-predictor-mdharris.streamlit.app](https://ames-house-price-predictor-mdharris.streamlit.app/)**

## Overview

This project walks through the full ML lifecycle — from raw data to a deployed, interpretable model:

1. **EDA & cleaning** on 79 explanatory variables describing residential homes in Ames, Iowa
2. **Feature engineering** — combination, interaction, binary-flag, and neighborhood-aggregate features
3. **Model development** — baseline through tuned stacking ensemble
4. **Interpretability** — SHAP analysis on the tree-based component, cross-checked against linear model coefficients
5. **Deployment** — a Streamlit app for interactive predictions with live SHAP explanations

## Results

| Metric | CV | Held-out Test |
|---|---|---|
| RMSLE | 0.1120 | **0.0990** |
| RMSE | — | **$20,653** |

The final model is a **stacking ensemble** (Ridge, Lasso, LightGBM, XGBoost base learners with a RidgeCV meta-learner). Test RMSLE beating CV RMSLE suggests the model generalizes well rather than overfitting to the training folds.

Random Forest and plain Linear Regression were tried but dropped from the final stack after showing no improvement in CV score.

## Approach

### Data cleaning
Systematic imputation guided by the data dictionary rather than blanket strategies — e.g. `NA` in fields like `BsmtQual` or `GarageType` means "feature absent," so those are imputed as `"None"`/`0` rather than dropped or mean-filled. `LotFrontage` is imputed using the most common value (mode) within each `MSZoning` group.

### Target and feature transforms
- `SalePrice` is right-skewed, so the model is trained on `log1p(SalePrice)` (which also matches Kaggle's own evaluation metric)
- Skewed numeric features (skew > 0.75) are log-transformed, *except* where a feature is dominated by zeros (>95%), which instead gets a binary presence flag to avoid distorting the distribution
- Rare categorical levels (<10 samples) are merged into an `"Other"` bucket to reduce overfitting risk

### Outlier handling
`GrLivArea` vs. `SalePrice` is used to identify the two well-documented outliers flagged in Dean De Cock's original paper (large homes sold at anomalously low prices — likely foreclosures or non-arm's-length sales). Ten additional 3σ deviations were identified but deliberately left in, to avoid over-pruning based on statistical outlier detection alone.

### Feature engineering
- **Combination features**: aggregating related columns (e.g. total square footage across floors and basement)
- **Interaction features**: `Qual_x_TotalSF`, `Qual_x_GrLivArea` — capturing joint effects neither feature expresses alone
- **Neighborhood aggregates**: safe, training-only target-style encoding of neighborhood price statistics

### Modeling
Two parallel preprocessing pipelines were built since linear and tree-based models need different treatment:
- **Linear models** (Ridge, Lasso): ordinal encoding for features with a meaningful order (e.g. quality ratings), target encoding for high-cardinality categoricals (>10 unique values), one-hot encoding for the remaining nominal features, plus scaling
- **Tree models** (XGBoost, LightGBM, Random Forest): ordinal encoding, no scaling

Models trained, in order: Linear Regression (baseline) → Ridge/Lasso (regularized) → Random Forest → XGBoost/LightGBM (Optuna-tuned) → Stacking ensemble (final).

Interestingly, dollar-scale RMSE on the test set told a different story than CV RMSLE: XGBoost and LightGBM were competitive on CV but underperformed on held-out dollar-scale error, because tree-based models can't extrapolate beyond the price range seen in training. The two largest XGBoost errors were both homes priced above $390K, outside the bulk of the training distribution — Lasso's linear extrapolation handled these better.

### Interpretability
Since SHAP doesn't cleanly support heterogeneous stacked ensembles, SHAP analysis is run on the XGBoost base component as a representative proxy, cross-validated against Lasso's coefficients (which used different encoding). Both model families agree on the top price drivers:

- Overall Quality
- Total square footage
- Above-grade living area
- Lot area
- House age
- Neighborhood

XGBoost leans heavily on the engineered interaction terms (especially `Qual_x_TotalSF`) over their raw components, while Lasso distributes importance more evenly and captures categorical signal better (owing to its one-hot/target encoding).

### Feature defaults for deployment
The model uses ~80+ raw features, so asking a user to fill in every one in the app would be impractical. Only the most impactful features (per the SHAP analysis) are exposed as interactive inputs; everything else defaults to a sensible value computed once from the training data — median for numeric features, mode for categorical — and saved to `models/feature_defaults.json`.

### Deployment
The trained stacking ensemble is served via a Streamlit app (`app.py`) that takes the highest-impact features as user input (in addition to defaults computed from training data for everything else) and returns a prediction with a live SHAP waterfall explanation.

**Serialization note**: the complete `StackingRegressor` isn't saved directly with `joblib.dump()`, because pickling the embedded XGBoost model across platforms (Linux training environment → local deployment) can corrupt its C++ binary state. Instead:
- Ridge, Lasso, and LightGBM pipelines are pickled normally
- The XGBoost model's weights are exported natively to JSON and reloaded into an XGBoost skeleton pipeline
- The stacking container is reconstructed manually at app startup, with the fitted estimators and internal routing parameters (`stack_method_`, `named_estimators_`) bound back in

## Project structure

```
.
├── EDA_FeatureEngg_Modeling_Analysis.ipynb       # Full EDA, feature engineering, modeling, SHAP analysis
├── app.py                                        # Streamlit prediction app
├── requirements.txt                              # Pinned dependency versions
├── runtime.txt                                   # Pinned Python version
├── preprocessing_utils.py                        # Custom transformer functions (must be importable for unpickling)
├── models/
│   ├── ridge_pipe.pkl
│   ├── lasso_pipe.pkl
│   ├── lgbm_pipe.pkl
│   ├── xgb_pipe_skeleton.pkl
│   ├── xgb_weights.json
│   ├── final_meta_learner.pkl
│   ├── feature_defaults.json
│   └── model_metadata.json
└── README.md
```

## Running locally

The app is [live on Streamlit Cloud](https://ames-house-price-predictor-mdharris.streamlit.app/), but to run it locally:

```bash
pip install -r requirements.txt

streamlit run app.py
```

The app loads all five model artifacts from `models/` and reconstructs the stacking ensemble at startup (cached via `@st.cache_resource`, so this only runs once per session).

## Tech stack

- **Modeling**: scikit-learn, XGBoost, LightGBM, Optuna (hyperparameter tuning)
- **Interpretability**: SHAP
- **Deployment**: Streamlit
- **Data**: pandas, NumPy

## Possible next steps

- Add input validation / range warnings in the app for out-of-distribution inputs (given the tree-model extrapolation limitation noted above)
- Extend SHAP explanation to a proper stacking-aware attribution (e.g. via `shap.KernelExplainer` on the full ensemble, at the cost of speed)
- Containerize the app (Dockerfile) for platform-independent deployment
