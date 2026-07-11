
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import pickle
import xgboost as xgb
from sklearn.ensemble import StackingRegressor

# ── MUST import before joblib.load() — pickle needs this ────────
from preprocessing_utils import log1p_dataframe

st.set_page_config(
    page_title="Ames House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ═══════════════════════════════════════════════════════════════
# Load model, defaults, and metadata (cached — runs once)
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    # 1. Load the perfectly healthy base pipelines
    with open("models/ridge_pipe.pkl", "rb") as f: ridge_p = pickle.load(f)
    with open("models/lasso_pipe.pkl", "rb") as f: lasso_p = pickle.load(f)
    with open("models/lgbm_pipe.pkl", "rb") as f: lgbm_p = pickle.load(f)
        
    # 2. Load the XGBoost skeleton pipeline and inject its native JSON weights
    with open("models/xgb_pipe_skeleton.pkl", "rb") as f: xgb_p = pickle.load(f)
    xgb_p.named_steps['model'].load_model("models/xgb_weights.json")
    
    # 3. Load the meta-learner
    with open("models/final_meta_learner.pkl", "rb") as f: meta_learner = pickle.load(f)
        
    # 4. Reconstruct the final Stacking Container natively on Windows
    compiled_stack = StackingRegressor(
        estimators=[('ridge', ridge_p), ('lasso', lasso_p), ('lgbm', lgbm_p), ('xgb', xgb_p)],
        final_estimator=meta_learner
    )
    
    # 5. Manually bind the fitted estimators AND internal stacking parameters
    compiled_stack.estimators_ = [ridge_p, lasso_p, lgbm_p, xgb_p]
    compiled_stack.final_estimator_ = meta_learner
    
    # Core Fix: Inject tracking parameters so scikit-learn knows how to route internal transforms
    compiled_stack.stack_method_ = ['predict'] * len(compiled_stack.estimators_)
    compiled_stack.named_estimators_ = {
        'ridge': ridge_p, 'lasso': lasso_p, 'lgbm': lgbm_p, 'xgb': xgb_p
    }
    
    return compiled_stack

@st.cache_resource
def load_defaults():
    with open("models/feature_defaults.json") as f:
        return json.load(f)

@st.cache_resource
def load_metadata():
    with open("models/model_metadata.json") as f:
        return json.load(f)

model     = load_model()
defaults  = load_defaults()
metadata  = load_metadata()

# ═══════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════
st.title("🏠 Ames House Price Predictor")
st.markdown(
    "Predicts house sale prices using a **stacked ensemble** "
    "(Ridge + Lasso + LightGBM + XGBoost) trained on the "
    "[Ames Housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)."
)

with st.expander("ℹ️ Model performance details"):
    st.write(f"**CV RMSLE:** {metadata['cv_rmsle']:.4f}")
    st.write(f"**Test RMSLE:** {metadata['test_rmsle']:.4f}")
    st.write(f"**Test RMSE:** ${metadata['test_rmse_dollars']:,.0f}")
    st.write(f"**Base models:** {', '.join(metadata['base_models'])}")

st.divider()

# ═══════════════════════════════════════════════════════════════
# User inputs — only the most impactful features (from SHAP)
# ═══════════════════════════════════════════════════════════════
st.subheader("📋 Enter House Details")

col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider(
        "Overall Quality (1=Poor, 10=Excellent)",
        min_value=1, max_value=10, value=6
    )
    gr_liv_area = st.number_input(
        "Above-Ground Living Area (sq ft)",
        min_value=300, max_value=6000, value=1500, step=50
    )
    total_bsmt_sf = st.number_input(
        "Total Basement Area (sq ft)",
        min_value=0, max_value=4000, value=900, step=50
    )
    year_built = st.number_input(
        "Year Built",
        min_value=1872, max_value=2010, value=1975, step=1
    )
    garage_cars = st.slider(
        "Garage Capacity (cars)",
        min_value=0, max_value=4, value=2
    )

with col2:
    lot_area = st.number_input(
        "Lot Area (sq ft)",
        min_value=1000, max_value=50000, value=9000, step=500
    )
    full_bath = st.slider("Full Bathrooms", 0, 4, 2)
    bedrooms  = st.slider("Bedrooms Above Grade", 0, 8, 3)
    neighborhood = st.selectbox(
        "Neighborhood",
        options=sorted([
            "NAmes", "CollgCr", "OldTown", "Edwards", "Somerst",
            "Gilbert", "NridgHt", "Sawyer", "NWAmes", "SawyerW",
            "BrkSide", "Crawfor", "Mitchel", "NoRidge", "Timber",
            "IDOTRR", "ClearCr", "StoneBr", "SWISU", "Blmngtn",
            "MeadowV", "BrDale", "Veenker", "NPkVill", "Blueste"
        ]),
        index=0
    )
    year_sold = st.number_input(
        "Year Sold", min_value=2006, max_value=2010, value=2008
    )

# ═══════════════════════════════════════════════════════════════
# Build full feature row — user inputs override defaults
# ═══════════════════════════════════════════════════════════════
def build_input_row(defaults, overrides):
    row = defaults.copy()
    row.update(overrides)
    return pd.DataFrame([row])

user_overrides = {
    "OverallQual" : overall_qual,
    "GrLivArea"   : gr_liv_area,
    "TotalBsmtSF" : total_bsmt_sf,
    "YearBuilt"   : year_built,
    "GarageCars"  : garage_cars,
    "LotArea"     : lot_area,
    "FullBath"    : full_bath,
    "BedroomAbvGr": bedrooms,
    "Neighborhood": neighborhood,
    "YrSold"      : year_sold,
}

# ── Recompute engineered features dependent on user inputs ──────
# (mirrors your Phase 3 feature engineering — keep in sync!)
user_overrides["HouseAge"] = user_overrides["YrSold"] - user_overrides["YearBuilt"]
user_overrides["TotalSF"]  = (
    user_overrides["TotalBsmtSF"] + defaults.get("1stFlrSF", 0) + defaults.get("2ndFlrSF", 0)
)
user_overrides["Qual_x_TotalSF"]   = user_overrides["OverallQual"] * user_overrides["TotalSF"]
user_overrides["Qual_x_GrLivArea"] = user_overrides["OverallQual"] * user_overrides["GrLivArea"]

input_df = build_input_row(defaults, user_overrides)

# ═══════════════════════════════════════════════════════════════
# Predict
# ═══════════════════════════════════════════════════════════════
st.divider()

if st.button("🔮 Predict Price", type="primary", use_container_width=True):
    with st.spinner("Predicting..."):
        pred_log   = model.predict(input_df)[0]
        pred_price = np.expm1(pred_log)

        st.success(f"### Predicted Price: **${pred_price:,.0f}**")

        # ── SHAP explanation using the XGBoost base component ────
        st.subheader("🔍 Why this prediction? (XGBoost component)")

        xgb_pipeline    = model.named_estimators_["xgb"]
        xgb_preprocessor = xgb_pipeline.named_steps["preprocessor"]
        xgb_model        = xgb_pipeline.named_steps["model"]

        input_processed = xgb_preprocessor.transform(input_df)

        explainer   = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(input_processed)

        explanation = shap.Explanation(
            values        = shap_values[0],
            base_values   = explainer.expected_value,
            data          = input_processed.iloc[0].values,
            feature_names = input_processed.columns.tolist()
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(explanation, max_display=10, show=False)
        st.pyplot(fig)

        st.caption(
            "This explanation reflects the XGBoost component of the "
            "stacking ensemble, used as a representative proxy since "
            "SHAP doesn't directly support heterogeneous stacked models."
        )

# ── Add a sidebar with project context (optional but nice touch) ──
with st.sidebar:
    st.header("About This Project")
    st.markdown("""
    This app demonstrates an end-to-end ML pipeline:
    - **EDA & feature engineering** on the Ames Housing dataset
    - **Stacking ensemble**: Ridge, Lasso, LightGBM, XGBoost
    - **Hyperparameter tuning** via Optuna
    - **SHAP interpretability** for model transparency

    [View full notebook on GitHub](https://github.com/mdharris128/ames-house-price-predictor/blob/main/EDA_FeatureEngg_Modeling_Analysis.ipynb)
    """)
    st.metric("Test RMSLE", f"{metadata['test_rmsle']:.4f}")
    st.metric("Test RMSE", f"${metadata['test_rmse_dollars']:,.0f}")

st.divider()
st.caption(
    "Built with scikit-learn, XGBoost, LightGBM, and SHAP | "
    "[View source on GitHub](https://github.com/mdharris128/ames-house-price-predictor)"
)
