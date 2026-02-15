import os
import hopsworks
import pandas as pd
import numpy as np
import joblib
import shutil
from datetime import datetime

from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

FG_NAME = "karachi_aqi_features_oct"
FG_VERSION = 1

MODEL_NAME = "1_year_champ"
MODEL_VERSION = 1 

TIMESTAMP_COL = "time"

TARGET_COLS = [
    "aqi_next_24h","aqi_next_48h","aqi_next_72h"
]

TRAIN_RATIO = 0.8

# Disable Arrow Flight (stability)
os.environ["HSFS_USE_ARROW_FLIGHT"] = "False"

# CONNECT TO HOPSWORKS
# =============================
print("\n🔌 Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()
mr = project.get_model_registry()

# LOAD FEATURE DATA
# =============================
print(f"\n📥 Loading Feature Group: {FG_NAME} v{FG_VERSION}")
fg = fs.get_feature_group(FG_NAME, FG_VERSION)
df = fg.read()
df=df.dropna()

df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

print(f"✅ Rows: {len(df)}")
print(f"🕒 Range: {df[TIMESTAMP_COL].min()} → {df[TIMESTAMP_COL].max()}")


# TEMPORAL SPLIT (NO GAP)
# =============================
split_idx = int(len(df) * TRAIN_RATIO)

train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

print(f"\n🔪 Temporal split")
print(f"   Train rows: {len(train_df)}")
print(f"   Test rows : {len(test_df)}")

# FEATURES & TARGETS
# =============================
X_train = train_df.drop(columns=TARGET_COLS + [TIMESTAMP_COL])
y_train = train_df[TARGET_COLS]

X_test = test_df.drop(columns=TARGET_COLS + [TIMESTAMP_COL])
y_test = test_df[TARGET_COLS]

print(f"\n📐 Shapes")
print(f"   X_train: {X_train.shape}")
print(f"   y_train: {y_train.shape}")
print(f"   X_test : {X_test.shape}")
print(f"   y_test : {y_test.shape}")


# MODELS (TUNED FOR AQI)
# =============================

models = {
    "LightGBM": MultiOutputRegressor(
        LGBMRegressor(
            n_estimators=400,        
            learning_rate=0.05,      
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            force_col_wise=True
        )
    ),
    "XGBoost": MultiOutputRegressor(
        XGBRegressor(
            n_estimators=400,        
            learning_rate=0.05,      
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1,                 
            reg_alpha=0.1,
            reg_lambda=1,
            objective="reg:squarederror",
            random_state=42,
            verbosity=0
        )
    ),
    "RandomForest": MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=500,        
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=4,      
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
    ),
}


# TRAIN & EVALUATE
# =============================
def evaluate(model, X, y):
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return rmse, mae, r2, y_pred

results = {}
predictions = {}

print("\n🚀 Training models...")

for name, model in models.items():
    print(f"\n🔧 {name}")
    model.fit(X_train, y_train)

    rmse, mae, r2, y_pred = evaluate(model, X_test, y_test)

    results[name] = {
        "model": model,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }
    predictions[name] = y_pred

    print(f"   RMSE: {rmse:.3f}")
    print(f"   MAE : {mae:.3f}")
    print(f"   R²  : {r2:.3f}")


# SELECT BEST MODEL
# =============================
best_name = min(results, key=lambda k: results[k]["rmse"])
best_model = results[best_name]["model"]
best_metrics = results[best_name]

print(f"\n🏆 BEST MODEL: {best_name}")
print(f"   RMSE: {best_metrics['rmse']:.3f}")
print(f"   MAE : {best_metrics['mae']:.3f}")
print(f"   R²  : {best_metrics['r2']:.3f}")


# PER-TARGET METRICS
# =============================
print("\n📊 Per-target metrics")
y_pred_best = predictions[best_name]

for i, target in enumerate(TARGET_COLS):
    rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred_best[:, i]))
    r2 = r2_score(y_test.iloc[:, i], y_pred_best[:, i])
    print(f"   {target}: RMSE={rmse:.3f}, R²={r2:.3f}")


# SAVE MODEL
# =============================
model_dir = "aqi_model_dir"
os.makedirs(model_dir, exist_ok=True)

joblib.dump(best_model, f"{model_dir}/model.pkl")

# Optional predictor
if os.path.exists("predictor.py"):
    shutil.copy("predictor.py", f"{model_dir}/predictor.py")


# REGISTER MODEL
# =============================
print("\n☁️ Registering model...")

metrics = {
    "rmse": float(best_metrics["rmse"]),
    "mae": float(best_metrics["mae"]),
    "r2": float(best_metrics["r2"]),
    "train_rows": len(train_df),
    "test_rows": len(test_df),
}


model_name_with_ts = f"{MODEL_NAME}"
model_registry = mr.python.create_model(
    name=model_name_with_ts,
    description=f"AQI multi-target model ({best_name}) – temporal split, shifted targets",
    metrics=metrics,
)

model_registry.save(model_dir)

print("\n✅ Training complete")
print(f"   Model: {model_name_with_ts}")
print(f"   Saved in: {model_dir}/")
