"""
Lahore AQI Forecasting Dashboard
Streamlit frontend — connects to the FastAPI inference backend
Run: streamlit run frontend/app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import json
import math
from pathlib import Path
from datetime import datetime, date, timedelta

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lahore AQI Forecast",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load models locally (no HTTP call needed) ─────────────────────────────────
BASE = Path(__file__).parent.parent
models_dir = BASE / "models"
data_dir   = BASE / "data"
plots_dir  = BASE / "plots"

@st.cache_resource
def load_models():
    lgb_reg = joblib.load(models_dir / "lgbm_regressor.pkl")
    lgb_cls = joblib.load(models_dir / "lgbm_classifier.pkl")
    xgb_reg = joblib.load(models_dir / "xgb_regressor.pkl")
    le      = joblib.load(models_dir / "label_encoder.pkl")
    with open(models_dir / "model_meta.json") as f:
        meta = json.load(f)
    return lgb_reg, lgb_cls, xgb_reg, le, meta

@st.cache_data
def load_dataset():
    df = pd.read_csv(data_dir / "lahore_aqi_dataset.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

lgb_reg, lgb_cls, xgb_reg, le, META = load_models()
df = load_dataset()

# ── AQI helpers ───────────────────────────────────────────────────────────────
CAT_COLORS = {
    "Good":                              "#00C853",
    "Moderate":                          "#D4A017",
    "Unhealthy for Sensitive Groups":    "#FF6D00",
    "Unhealthy":                         "#DD2C00",
    "Very Unhealthy":                    "#6200EA",
    "Hazardous":                         "#37474F",
}
CAT_ADVISORY = {
    "Good":                              "Air is clean. Great day for outdoor activities!",
    "Moderate":                          "Acceptable. Unusually sensitive people may want to limit prolonged outdoor exertion.",
    "Unhealthy for Sensitive Groups":    "Children, elderly, and people with respiratory/heart conditions should reduce outdoor activity.",
    "Unhealthy":                         "Everyone may begin to feel health effects. Limit prolonged outdoor exertion.",
    "Very Unhealthy":                    "Health alert! Avoid prolonged outdoor exertion. Wear N95 mask if going outside.",
    "Hazardous":                         "HEALTH EMERGENCY. Avoid ALL outdoor activity. Stay indoors with windows closed.",
}

def aqi_to_pm25(aqi):
    if aqi<=50:    return aqi*12/50
    elif aqi<=100: return 12+(aqi-50)*23.4/50
    elif aqi<=150: return 35.4+(aqi-100)*20/50
    elif aqi<=200: return 55.4+(aqi-150)*54.3/50
    elif aqi<=300: return 150.4+(aqi-200)*99.9/100
    else:          return 250.4+(aqi-300)*149.9/200

def predict_aqi(feat_vec):
    X = np.array(feat_vec, dtype=float).reshape(1, -1)
    pred_lgb = float(lgb_reg.predict(X)[0])
    pred_xgb = float(xgb_reg.predict(X)[0])
    pred = pred_lgb * 0.4 + pred_xgb * 0.6
    pred = max(0, min(600, pred))
    proba = lgb_cls.predict_proba(X)[0]
    cls_idx = int(np.argmax(proba))
    cat = le.classes_[cls_idx]
    conf = float(proba[cls_idx]) * 100
    return round(pred, 1), cat, round(conf, 1)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.aqi-card {
    padding: 1.2rem 1.5rem;
    border-radius: 18px;
    text-align: center;
    min-height: 170px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 18px 30px rgba(4, 11, 32, 0.12);
}
.aqi-number { font-size: 3.5rem; font-weight: 700; line-height: 1.1; }
.aqi-label  { font-size: 1.1rem; font-weight: 600; margin-top: 4px; }
.metric-box,
.forecast-panel {
    background: #f7f8fb;
    border-radius: 18px;
    padding: 1.1rem 1rem;
    border: 1px solid rgba(0,0,0,0.08);
    min-height: 170px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 0.35rem;
    box-shadow: 0 10px 18px rgba(0, 0, 0, 0.05);
}
.forecast-row {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 1rem;
    align-items: start;
    margin-top: 1rem;
}
.forecast-value { font-size: 1.6rem; font-weight: 700; color: #111; }
.forecast-label { color: #555; font-size: 0.95rem; margin-top: 0.25rem; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🌫️ Lahore AQI Forecast")
    st.markdown("---")

    today_aqi = float(df["aqi"].iloc[-1])
    today_cat = df["aqi_category"].iloc[-1]
    today_color = CAT_COLORS[today_cat]

    st.markdown(f"""
    <div class="aqi-card" style="background:{today_color}22; border:1.5px solid {today_color};">
        <div style="font-size:0.8rem; color:#666;">Latest recorded AQI</div>
        <div class="aqi-number" style="color:{today_color};">{today_aqi:.0f}</div>
        <div class="aqi-label" style="color:{today_color}; font-size:0.85rem;">{today_cat}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Dataset:** 365 days")
    st.markdown("**Training set:** 285 days")
    st.markdown("**Test set:** 72 days")
    st.markdown("**Features:** 27")
    st.markdown(f"**Best MAE:** {META['metrics']['xgb_reg_mae']}")
    st.markdown("---")
    st.markdown("**AQI Scale (US EPA)**")
    for cat, color in CAT_COLORS.items():
        short = cat.replace("Unhealthy for Sensitive Groups", "USG")
        st.markdown(f'<span style="color:{color};">■</span> {short}', unsafe_allow_html=True)

st.markdown("## 🌫️ Lahore AQI Next-Day Forecasting")
st.markdown("**ML-powered air quality prediction** | LightGBM + XGBoost Ensemble | Trained on 365 days of Lahore data")
st.divider()

# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict Tomorrow", "📊 Data Explorer", "🤖 Model Performance", "ℹ️ About"])

# ─────────────────────────────────────────────────────────────────────
# TAB 1: PREDICT
# ─────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Enter today's conditions to forecast tomorrow's AQI")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### 📅 Recent AQI readings")
        aqi_today     = st.number_input("Today's AQI",      min_value=0.0, max_value=600.0, value=185.0, step=1.0)
        aqi_yesterday = st.number_input("Yesterday's AQI",  min_value=0.0, max_value=600.0, value=195.0, step=1.0)
        aqi_2d        = st.number_input("2 days ago",        min_value=0.0, max_value=600.0, value=178.0, step=1.0)
        aqi_6d        = st.number_input("6 days ago",        min_value=0.0, max_value=600.0, value=160.0, step=1.0)
        aqi_7d        = st.number_input("7 days ago",        min_value=0.0, max_value=600.0, value=155.0, step=1.0)
        pm25_today    = st.number_input("Today's PM2.5 (µg/m³)", min_value=0.0, max_value=500.0, value=92.0, step=1.0)
        pm25_3d       = st.number_input("PM2.5 — 3 days ago", min_value=0.0, max_value=500.0, value=88.0, step=1.0)

    with col_right:
        st.markdown("#### 🌤️ Tomorrow's weather forecast")
        temp_c      = st.slider("Temperature (°C)", -5.0, 50.0, 15.0, 0.5)
        humidity    = st.slider("Humidity (%)",       0.0, 100.0, 75.0, 1.0)
        wind_speed  = st.slider("Wind Speed (m/s)",   0.0, 15.0, 2.0, 0.1)
        wind_dir    = st.slider("Wind Direction (°)", 0.0, 360.0, 280.0, 5.0)
        precip      = st.slider("Precipitation (mm)", 0.0, 50.0, 0.0, 0.5)

        st.markdown("#### 🗓️ Context")
        col_a, col_b = st.columns(2)
        with col_a:
            tomorrow = st.date_input("Date of forecast", value=date.today() + timedelta(days=1))
            month = tomorrow.month
            dow   = tomorrow.weekday()
        with col_b:
            is_fog = st.checkbox("Fog season (Nov–Feb)", value=month in [11,12,1,2])
            is_wk  = st.checkbox("Weekend", value=dow >= 5)
            is_ram = st.checkbox("Ramadan", value=False)

    # ── Compute rolling stats from inputs ─────────────────────────────────────
    recent = [aqi_today, aqi_yesterday, aqi_2d, aqi_6d, aqi_7d]
    roll_mean_3 = round(np.mean([aqi_today, aqi_yesterday, aqi_2d]), 1)
    roll_mean_7 = round(np.mean(recent + [160.0, 162.0]), 1)
    roll_std_7  = round(float(np.std(recent + [160.0, 162.0])), 1)
    roll_max_3  = round(max(aqi_today, aqi_yesterday, aqi_2d), 1)

    wind_x = wind_speed * math.cos(math.radians(wind_dir))
    wind_y = wind_speed * math.sin(math.radians(wind_dir))
    heat_index = temp_c * (1 + 0.0055 * (humidity - 58))
    inversion  = int(temp_c < 15 and humidity > 65)
    dispersion = wind_speed / (humidity / 100 + 0.1)
    month_sin  = math.sin(2 * math.pi * month / 12)
    month_cos  = math.cos(2 * math.pi * month / 12)
    dow_sin    = math.sin(2 * math.pi * dow / 7)
    dow_cos    = math.cos(2 * math.pi * dow / 7)

    feat_vec = [
        aqi_today, aqi_yesterday, aqi_2d, aqi_6d, aqi_7d,
        roll_mean_3, roll_mean_7, roll_std_7, roll_max_3,
        pm25_today, pm25_3d,
        temp_c, humidity, wind_speed, precip,
        wind_x, wind_y, heat_index, inversion, dispersion,
        month_sin, month_cos, dow_sin, dow_cos,
        int(is_fog), int(is_wk), int(is_ram),
    ]

    st.divider()
    if st.button("🔮 Predict Tomorrow's AQI", type="primary", use_container_width=True):
        pred_aqi, category, confidence = predict_aqi(feat_vec)
        pm25_est = round(aqi_to_pm25(pred_aqi), 1)
        color = CAT_COLORS[category]
        advisory = CAT_ADVISORY[category]

        st.markdown("### 📍 Forecast Result")
        st.markdown(f"""
        <div class="forecast-row">
            <div class="aqi-card" style="background:{color}22; border: 2px solid {color};">
                <div class="aqi-number" style="color:{color};">{pred_aqi}</div>
                <div class="aqi-label" style="color:{color};">AQI</div>
            </div>
            <div class="forecast-panel">
                <div class="forecast-value">{pm25_est}</div>
                <div class="forecast-label">PM2.5 (µg/m³)</div>
            </div>
            <div class="forecast-panel">
                <div class="forecast-value" style="color:{color};">{category}</div>
                <div class="forecast-label">AQI category</div>
            </div>
            <div class="forecast-panel">
                <div class="forecast-value">{confidence}%</div>
                <div class="forecast-label">Model confidence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:{color}18; border-left:4px solid {color}; padding:1rem 1.2rem; border-radius:4px; margin:1rem 0;">
            <strong>⚕️ Health Advisory:</strong> {advisory}
        </div>""", unsafe_allow_html=True)

        st.info(f"Model MAE: ±{META['metrics']['xgb_reg_mae']} AQI units  |  "
                f"Ensemble: 60% XGBoost + 40% LightGBM  |  "
                f"27 input features  |  Trained on 285 days")

# ─────────────────────────────────────────────────────────────────────
# TAB 2: DATA EXPLORER
# ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📊 Lahore AQI Data Explorer — April 2024 to April 2025")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean AQI",  f"{df['aqi'].mean():.1f}")
    c2.metric("Max AQI",   f"{df['aqi'].max():.0f}")
    c3.metric("Hazardous days", f"{(df['aqi_category']=='Hazardous').sum()}")
    c4.metric("Good days",      f"{(df['aqi_category']=='Good').sum()}")

    # Time series plot
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(df["date"], df["aqi"], color="#1565C0", linewidth=0.9, alpha=0.85)
    ax.fill_between(df["date"], df["aqi"], alpha=0.1, color="#1565C0")
    for thresh, col, lbl in [(50,"#00C853","Good"),(100,"#FFD600","Moderate"),
                              (200,"#DD2C00","Unhealthy"),(300,"#6200EA","Very Unhealthy")]:
        ax.axhline(thresh, linestyle="--", linewidth=0.7, color=col, label=lbl)
    ax.set_title("Daily AQI — Lahore 2024-25", fontsize=12, fontweight="bold")
    ax.set_ylabel("AQI"); ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    col_a, col_b = st.columns(2)
    with col_a:
        st.image(str(plots_dir / "02_monthly_aqi.png"), caption="Monthly Average AQI", use_container_width=True)
    with col_b:
        st.image(str(plots_dir / "03_correlation_heatmap.png"), caption="Feature Correlation Heatmap", use_container_width=True)

    st.markdown("#### Raw dataset (last 30 days)")
    st.dataframe(df.tail(30)[["date","aqi","aqi_category","pm25_ugm3","temp_c",
                               "humidity_pct","wind_speed_ms","precipitation_mm",
                               "is_fog_season"]].sort_values("date",ascending=False),
                 use_container_width=True, height=320)

# ─────────────────────────────────────────────────────────────────────
# TAB 3: MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 🤖 Model Training Results")

    m = META["metrics"]
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("XGBoost MAE",    f"{m['xgb_reg_mae']:.1f}", help="Mean Absolute Error (AQI units)")
    mc2.metric("XGBoost RMSE",   f"{m['xgb_reg_rmse']:.1f}")
    mc3.metric("LightGBM MAE",   f"{m['lgbm_reg_mae']:.1f}")
    mc4.metric("LightGBM RMSE",  f"{m['lgbm_reg_rmse']:.1f}")
    mc5.metric("Classifier F1",  f"{m['lgbm_cls_f1']:.3f}", help="Weighted F1 (6-class)")

    p1, p2 = st.columns(2)
    with p1:
        st.image(str(plots_dir / "05_predicted_vs_actual.png"),
                 caption="Predicted vs Actual AQI (Test Set)", use_container_width=True)
    with p2:
        st.image(str(plots_dir / "06_confusion_matrix.png"),
                 caption="Confusion Matrix — AQI Category", use_container_width=True)

    st.image(str(plots_dir / "04_shap_importance.png"),
             caption="SHAP Feature Importance Analysis", use_container_width=True)

    st.markdown("""
    **Interpretation:**
    - `aqi_lag_1` (yesterday's AQI) is the single strongest predictor — air quality has strong inertia
    - `is_fog_season` and `inversion_risk` capture Lahore's winter smog phenomenon
    - `wind_speed_ms` shows strong negative SHAP values — higher wind = lower next-day AQI
    - `precipitation_mm` has outsized impact on days it occurs — rain scrubs PM2.5
    """)

# ─────────────────────────────────────────────────────────────────────
# TAB 4: ABOUT
# ─────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("""
    ### About This Project

    **Lahore AQI Next-Day Forecasting** is an end-to-end ML project that predicts the following day's
    Air Quality Index for Lahore, Pakistan — one of the world's most polluted cities.

    #### Why Lahore?
    - Ranked among the top 5 most polluted cities globally (IQAir 2023)
    - Annual average PM2.5: ~97 µg/m³ (WHO guideline: 5 µg/m³)
    - Severe "smog season" every November–February due to temperature inversions
    - No existing hyperlocal ML-based next-day forecast

    #### Data Sources
    | Source | Data | Update |
    |--------|------|--------|
    | WAQI API (waqi.info) | PM2.5, PM10, O3, NO2, CO | Hourly |
    | Open-Meteo API | Temperature, humidity, wind, precipitation | Hourly |
    | Manual / calendar | Fog season flag, Ramadan, weekends | Daily |

    #### Models
    | Model | Task | MAE | F1 |
    |-------|------|-----|-----|
    | XGBoost | Regression | 49.0 | — |
    | LightGBM | Regression + Classification | 55.7 | 0.309 |
    | **Ensemble** | **Final prediction** | **~47** | — |

    #### Feature Engineering
    27 features including 5 AQI lag features, 4 rolling statistics, 5 weather variables,
    3 weather interaction terms, 4 cyclical time encodings, and 3 context flags.

    #### API
    ```
    POST /predict
    Returns: predicted_aqi, aqi_category, confidence_pct, pm25_estimate, health_advisory
    ```

    #### Future Work
    - Multi-station data (more Lahore monitoring points)
    - Traffic density integration (Open Street Map)
    - SMS alert system for hazardous days
    - Integration with Pak-EPA official data
    - Hourly granularity (currently daily)

    ---
    Built with LightGBM · XGBoost · SHAP · FastAPI · Streamlit
    """)

