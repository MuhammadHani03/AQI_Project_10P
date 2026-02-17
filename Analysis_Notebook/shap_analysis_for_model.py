"""
SHAP Analysis for 24h AQI Forecast
Features: numeric only
Target: aqi_next_24h
Timestamp column dropped
"""

# ==============================
# IMPORTS
# ==============================
import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import hopsworks
from dotenv import load_dotenv

# ==============================
# CONFIG
# ==============================
load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
PROJECT_ID = None  # Optional: set project ID
FG_NAME = "karachi_aqi_features_oct"
FG_VERSION = 1
MODEL_NAME = "1_year_champ"

TARGET_COL = "aqi_next_24h"
TIMESTAMP_COL = "time"
SHAP_OUTPUT_DIR = "shap_output"
os.makedirs(SHAP_OUTPUT_DIR, exist_ok=True)

# ==============================
# 1️⃣ Login & Get Hopsworks objects
# ==============================
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=PROJECT_ID)
fs = project.get_feature_store()
mr = project.get_model_registry()

# ==============================
# 2️⃣ Load latest model
# ==============================
print(f"📥 Loading model '{MODEL_NAME}' from Hopsworks...")
model_obj = mr.get_best_model(name=MODEL_NAME, metric="rmse", direction="min")
model_dir = model_obj.download()
model_path = os.path.join(model_dir, "model.pkl")
model = joblib.load(model_path)
print("✅ Model loaded:", type(model))

# Use first estimator if MultiOutputRegressor
if hasattr(model, "estimators_"):
    base_model = model.estimators_[0]
else:
    base_model = model

# ==============================
# 3️⃣ Load feature group from Hopsworks
# ==============================
print(f"📥 Loading Feature Group '{FG_NAME}' v{FG_VERSION} ...")
fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)
df = fg.read()
df = df.dropna()
df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

# Drop timestamp and target
X_features = df.drop(columns=[TIMESTAMP_COL, TARGET_COL])
X_features = X_features.select_dtypes(include=[np.number])

# Align features with model
if hasattr(base_model, "feature_names_in_"):
    X_features = X_features[base_model.feature_names_in_]
else:
    X_features = X_features.copy()

X_transformed = X_features.values
print("Final feature matrix shape:", X_transformed.shape)

# ==============================
# 4️⃣ SHAP Analysis
# ==============================
explainer = shap.TreeExplainer(base_model)
shap_values = explainer.shap_values(X_transformed)
print("✅ SHAP values computed, shape:", shap_values.shape)

# ==============================
# 5️⃣ SHAP summary plots
# ==============================
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_features, show=False)
plt.tight_layout()
plt.savefig(os.path.join(SHAP_OUTPUT_DIR, "shap_summary_24h.png"), dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_features, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(SHAP_OUTPUT_DIR, "shap_bar_24h.png"), dpi=300, bbox_inches='tight')
plt.close()

# ==============================
# 6️⃣ Optional: Force plot for first sample
# ==============================
plt.figure()
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_features.iloc[0],
    matplotlib=True,
    show=False
)
plt.tight_layout()
plt.savefig(os.path.join(SHAP_OUTPUT_DIR, "force_plot_24h_sample.png"), dpi=300, bbox_inches='tight')
plt.close()

# ==============================
# 7️⃣ Mean absolute SHAP per feature
# ==============================
importance = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    "feature": X_features.columns,
    "mean_abs_shap": importance
}).sort_values("mean_abs_shap", ascending=False)

print("\n✅ SHAP analysis complete. Top features by importance:")
print(importance_df.head(15))
