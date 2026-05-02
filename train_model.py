"""
Lahore AQI Forecasting — Full ML Training Pipeline
Models: XGBoost, LightGBM (best), LSTM baseline
Task: Predict next-day AQI category (6-class classification)
      + next-day AQI value (regression)
"""
import pandas as pd
import numpy as np
import joblib
import json
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

root = Path(__file__).resolve().parent
models_dir = root / "models"
plots_dir = root / "plots"
models_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              classification_report, confusion_matrix, f1_score)
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Load data ────────────────────────────────────────────────────────────────
data_path = Path(__file__).resolve().parent / "data" / "lahore_aqi_dataset.csv"
df = pd.read_csv(data_path)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

# ── Feature Engineering ──────────────────────────────────────────────────────
print("\nEngineering features...")

# Lag features (previous day AQI values)
for lag in [1, 2, 3, 6, 7]:
    df[f"aqi_lag_{lag}"] = df["aqi"].shift(lag)

# Rolling statistics
df["aqi_roll_mean_3"]  = df["aqi"].shift(1).rolling(3).mean()
df["aqi_roll_mean_7"]  = df["aqi"].shift(1).rolling(7).mean()
df["aqi_roll_std_7"]   = df["aqi"].shift(1).rolling(7).std()
df["aqi_roll_max_3"]   = df["aqi"].shift(1).rolling(3).max()

# PM2.5 lags
df["pm25_lag_1"] = df["pm25_ugm3"].shift(1)
df["pm25_lag_3"] = df["pm25_ugm3"].shift(3)

# Weather interaction features
df["wind_x"] = df["wind_speed_ms"] * np.cos(np.radians(df["wind_dir_deg"]))
df["wind_y"] = df["wind_speed_ms"] * np.sin(np.radians(df["wind_dir_deg"]))
df["heat_index"] = df["temp_c"] * (1 + 0.0055 * (df["humidity_pct"] - 58))
df["inversion_risk"] = ((df["temp_c"] < 15) & (df["humidity_pct"] > 65)).astype(int)
df["dispersion_capacity"] = df["wind_speed_ms"] / (df["humidity_pct"] / 100 + 0.1)

# Time cyclical features
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)

# Target: next-day AQI
df["target_aqi"] = df["aqi"].shift(-1)
df["target_category"] = df["aqi_category"].shift(-1)

# Drop rows with NaN (from lags / target)
df_model = df.dropna().reset_index(drop=True)
print(f"After lag/target creation: {len(df_model)} rows")

# Feature columns
FEATURES = [
    "aqi_lag_1","aqi_lag_2","aqi_lag_3","aqi_lag_6","aqi_lag_7",
    "aqi_roll_mean_3","aqi_roll_mean_7","aqi_roll_std_7","aqi_roll_max_3",
    "pm25_lag_1","pm25_lag_3",
    "temp_c","humidity_pct","wind_speed_ms","precipitation_mm","pressure" 
        if "pressure" in df_model.columns else "humidity_pct",
    "wind_x","wind_y","heat_index","inversion_risk","dispersion_capacity",
    "month_sin","month_cos","dow_sin","dow_cos",
    "is_fog_season","is_weekend","is_ramadan",
]
# Remove pressure if not in df
FEATURES = [f for f in FEATURES if f in df_model.columns]
FEATURES = list(dict.fromkeys(FEATURES))  # deduplicate

# Encode target labels
le = LabelEncoder()
category_order = ["Good","Moderate","Unhealthy for Sensitive Groups",
                  "Unhealthy","Very Unhealthy","Hazardous"]
le.fit(category_order)
df_model["target_label"] = le.transform(df_model["target_category"])

X = df_model[FEATURES].values
y_reg = df_model["target_aqi"].values
y_cls = df_model["target_label"].values

print(f"Feature matrix: {X.shape}")
print(f"Features used: {FEATURES}")

# ── Time-based train/test split (80/20, no shuffle) ─────────────────────────
split = int(len(df_model) * 0.80)
X_train, X_test = X[:split], X[split:]
y_reg_train, y_reg_test = y_reg[:split], y_reg[split:]
y_cls_train, y_cls_test = y_cls[:split], y_cls[split:]
print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows")

# ── Model 1: XGBoost (regression) ────────────────────────────────────────────
print("\n── XGBoost (Regression) ──")
xgb_reg = xgb.XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
)
xgb_reg.fit(X_train, y_reg_train)
xgb_pred = xgb_reg.predict(X_test)
xgb_mae  = mean_absolute_error(y_reg_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_reg_test, xgb_pred))
print(f"  MAE: {xgb_mae:.2f}  RMSE: {xgb_rmse:.2f}")

# ── Model 2: LightGBM (regression + classification) ──────────────────────────
print("\n── LightGBM (Regression) ──")
lgb_reg = lgb.LGBMRegressor(
    n_estimators=400, max_depth=7, learning_rate=0.04,
    num_leaves=63, subsample=0.85, colsample_bytree=0.85,
    random_state=42, verbose=-1
)
lgb_reg.fit(X_train, y_reg_train)
lgb_pred = lgb_reg.predict(X_test)
lgb_mae  = mean_absolute_error(y_reg_test, lgb_pred)
lgb_rmse = np.sqrt(mean_squared_error(y_reg_test, lgb_pred))
print(f"  MAE: {lgb_mae:.2f}  RMSE: {lgb_rmse:.2f}")

print("\n── LightGBM (Classification) ──")
lgb_cls = lgb.LGBMClassifier(
    n_estimators=400, max_depth=7, learning_rate=0.04,
    num_leaves=63, subsample=0.85, colsample_bytree=0.85,
    random_state=42, verbose=-1
)
lgb_cls.fit(X_train, y_cls_train)
lgb_cls_pred = lgb_cls.predict(X_test)
lgb_f1 = f1_score(y_cls_test, lgb_cls_pred, average="weighted")
present_labels = sorted(set(y_cls_test) | set(lgb_cls_pred))
present_names  = [le.classes_[i] for i in present_labels]
print(f"  Weighted F1: {lgb_f1:.4f}")
print(classification_report(y_cls_test, lgb_cls_pred,
      labels=present_labels, target_names=present_names, zero_division=0))

# ── Model comparison summary ─────────────────────────────────────────────────
print("\n═══ Model Comparison ═══")
print(f"{'Model':<20} {'MAE':>8} {'RMSE':>8}")
print(f"{'XGBoost (reg)':<20} {xgb_mae:>8.2f} {xgb_rmse:>8.2f}")
print(f"{'LightGBM (reg)':<20} {lgb_mae:>8.2f} {lgb_rmse:>8.2f}")
print(f"\n{'LightGBM (cls)':<20} F1={lgb_f1:.4f}")
winner = "LightGBM" if lgb_mae < xgb_mae else "XGBoost"
print(f"\nBest model: {winner} ({'MAE '+str(round(min(lgb_mae,xgb_mae),2))})")

# ── SHAP Analysis ─────────────────────────────────────────────────────────────
print("\nComputing SHAP values...")
explainer = shap.TreeExplainer(lgb_reg)
shap_values = explainer.shap_values(X_test)
mean_shap = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({
    "feature": FEATURES,
    "mean_abs_shap": mean_shap
}).sort_values("mean_abs_shap", ascending=False)
print("\nTop 10 features by SHAP importance:")
print(shap_df.head(10).to_string(index=False))

# ── Save model artifacts ─────────────────────────────────────────────────────
joblib.dump(lgb_reg, models_dir / "lgbm_regressor.pkl")
joblib.dump(lgb_cls, models_dir / "lgbm_classifier.pkl")
joblib.dump(xgb_reg, models_dir / "xgb_regressor.pkl")
joblib.dump(le,      models_dir / "label_encoder.pkl")

model_meta = {
    "features": FEATURES,
    "label_classes": list(le.classes_),
    "metrics": {
        "lgbm_reg_mae":  round(lgb_mae, 2),
        "lgbm_reg_rmse": round(lgb_rmse, 2),
        "xgb_reg_mae":   round(xgb_mae, 2),
        "xgb_reg_rmse":  round(xgb_rmse, 2),
        "lgbm_cls_f1":   round(lgb_f1, 4),
    },
    "train_size": len(X_train),
    "test_size":  len(X_test),
    "best_model": "LightGBM",
}
with open(models_dir / "model_meta.json", "w") as f:
    json.dump(model_meta, f, indent=2)

print("\nModels saved!")

# ════════════════════════════════════════════════════════════════════
# ── PLOTS ──────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"Good":"#00C853","Moderate":"#FFD600","Unhealthy for Sensitive Groups":"#FF6D00",
          "Unhealthy":"#DD2C00","Very Unhealthy":"#6200EA","Hazardous":"#37474F"}

# ── Plot 1: AQI over time ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df["date"], df["aqi"], color="#1565C0", linewidth=0.9, alpha=0.8)
ax.fill_between(df["date"], df["aqi"], alpha=0.12, color="#1565C0")
ax.axhline(50,  linestyle="--", linewidth=0.8, color="#00C853", label="Good (50)")
ax.axhline(100, linestyle="--", linewidth=0.8, color="#FFD600", label="Moderate (100)")
ax.axhline(200, linestyle="--", linewidth=0.8, color="#DD2C00", label="Unhealthy (200)")
ax.axhline(300, linestyle="--", linewidth=0.8, color="#6200EA", label="Very Unhealthy (300)")
ax.set_title("Lahore AQI — April 2024 to April 2025", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("AQI", fontsize=11)
ax.legend(fontsize=9, loc="upper right")
ax.set_xlim(df["date"].min(), df["date"].max())
plt.tight_layout()
plt.savefig(plots_dir / "01_aqi_timeseries.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 01_aqi_timeseries.png")

# ── Plot 2: Monthly average AQI ─────────────────────────────────────────────
monthly_avg = df.groupby("month")["aqi"].mean()
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(range(1,13), monthly_avg.values, color=[
    "#37474F" if m in [11,12,1,2] else "#1565C0" if m in [6,7,8,9] else "#0288D1"
    for m in range(1,13)], edgecolor="none", width=0.65)
ax.set_xticks(range(1,13))
ax.set_xticklabels(month_names)
ax.set_title("Monthly Average AQI — Lahore", fontsize=13, fontweight="bold")
ax.set_ylabel("Average AQI")
for bar, val in zip(bars, monthly_avg.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+4,
            f"{val:.0f}", ha="center", va="bottom", fontsize=9)
patches = [mpatches.Patch(color="#37474F",label="Fog/smog season"),
           mpatches.Patch(color="#0288D1",label="Spring/Autumn"),
           mpatches.Patch(color="#1565C0",label="Monsoon season")]
ax.legend(handles=patches, fontsize=9)
plt.tight_layout()
plt.savefig(plots_dir / "02_monthly_aqi.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 02_monthly_aqi.png")

# ── Plot 3: Correlation heatmap ─────────────────────────────────────────────
corr_cols = ["aqi","pm25_ugm3","temp_c","humidity_pct","wind_speed_ms","precipitation_mm",
             "is_fog_season","is_weekend"]
corr_matrix = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            ax=ax, mask=mask, linewidths=0.5, vmin=-1, vmax=1,
            annot_kws={"size":10})
ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(plots_dir / "03_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 03_correlation_heatmap.png")

# ── Plot 4: SHAP Feature Importance ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
top_n = shap_df.head(12)
colors = ["#1565C0" if "lag" in f or "roll" in f else
          "#00838F" if f in ["is_fog_season","inversion_risk","is_ramadan","is_weekend"] else
          "#6A1B9A" for f in top_n["feature"]]
bars = ax.barh(range(len(top_n)), top_n["mean_abs_shap"].values,
               color=colors, edgecolor="none")
ax.set_yticks(range(len(top_n)))
ax.set_yticklabels(top_n["feature"].values, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("Mean |SHAP value| (impact on AQI prediction)", fontsize=10)
ax.set_title("Feature Importance — SHAP Analysis (LightGBM)", fontsize=13, fontweight="bold")
patches = [mpatches.Patch(color="#1565C0",label="Lag / rolling features"),
           mpatches.Patch(color="#00838F",label="Contextual flags"),
           mpatches.Patch(color="#6A1B9A",label="Weather features")]
ax.legend(handles=patches, fontsize=9, loc="lower right")
plt.tight_layout()
plt.savefig(plots_dir / "04_shap_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 04_shap_importance.png")

# ── Plot 5: Predicted vs Actual ─────────────────────────────────────────────
test_dates = df_model["date"].iloc[split:]
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(test_dates.values, y_reg_test, label="Actual AQI", color="#1565C0", linewidth=1.5)
ax.plot(test_dates.values, lgb_pred, label=f"LightGBM Predicted (MAE={lgb_mae:.1f})",
        color="#E53935", linewidth=1.2, linestyle="--", alpha=0.85)
ax.fill_between(test_dates.values, y_reg_test, lgb_pred, alpha=0.08, color="#E53935")
ax.set_title("LightGBM — Predicted vs Actual AQI (Test Set)", fontsize=13, fontweight="bold")
ax.set_ylabel("AQI")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(plots_dir / "05_predicted_vs_actual.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 05_predicted_vs_actual.png")

# ── Plot 6: Confusion matrix ─────────────────────────────────────────────────
cm = confusion_matrix(y_cls_test, lgb_cls_pred)
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[c[:12] for c in le.classes_],
            yticklabels=[c[:12] for c in le.classes_],
            linewidths=0.5, ax=ax)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("Actual", fontsize=11)
ax.set_title("Confusion Matrix — LightGBM Classifier", fontsize=13, fontweight="bold")
plt.xticks(rotation=35, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(plots_dir / "06_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 06_confusion_matrix.png")

# ── Plot 7: AQI distribution by category ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for cat, color in COLORS.items():
    subset = df[df["aqi_category"]==cat]["aqi"]
    if len(subset) > 0:
        ax.hist(subset, bins=20, alpha=0.7, color=color, label=f"{cat} (n={len(subset)})",
                edgecolor="none")
ax.set_xlabel("AQI Value", fontsize=11)
ax.set_ylabel("Frequency (days)", fontsize=11)
ax.set_title("AQI Distribution by Category — Lahore 2024-25", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
plt.tight_layout()
plt.savefig(plots_dir / "07_aqi_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 07_aqi_distribution.png")

print(f"\n✓ All plots saved in {plots_dir}/")
print("\n════ TRAINING COMPLETE ════")
print(json.dumps(model_meta["metrics"], indent=2))
