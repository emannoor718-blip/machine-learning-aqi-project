"""
Lahore AQI Forecasting — FastAPI Inference API
Endpoint: POST /predict
Returns: next-day AQI prediction + category + confidence + health advisory
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import joblib
import json
import math
from pathlib import Path

# ── Load models ───────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
models_dir = BASE / "models"

lgb_reg  = joblib.load(models_dir / "lgbm_regressor.pkl")
lgb_cls  = joblib.load(models_dir / "lgbm_classifier.pkl")
xgb_reg  = joblib.load(models_dir / "xgb_regressor.pkl")
le       = joblib.load(models_dir / "label_encoder.pkl")

with open(models_dir / "model_meta.json") as f:
    META = json.load(f)

FEATURES = META["features"]

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Lahore AQI Forecasting API",
    description="Next-day AQI prediction for Lahore using LightGBM + XGBoost models trained on Lahore weather and historical AQI data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response schemas ────────────────────────────────────────────────
class AQIInput(BaseModel):
    # Yesterday's AQI readings (required for lag features)
    aqi_today: float          = Field(..., ge=0, le=600, description="Today's AQI", example=185.0)
    aqi_yesterday: float      = Field(..., ge=0, le=600, example=195.0)
    aqi_2days_ago: float      = Field(..., ge=0, le=600, example=178.0)
    aqi_6days_ago: float      = Field(..., ge=0, le=600, example=160.0)
    aqi_7days_ago: float      = Field(..., ge=0, le=600, example=155.0)
    pm25_today: float         = Field(..., ge=0, le=500, description="Today's PM2.5 µg/m³", example=95.0)
    pm25_3days_ago: float     = Field(..., ge=0, le=500, example=88.0)

    # Rolling stats (pass pre-computed or the API can compute from lags)
    aqi_roll_mean_3: float    = Field(..., example=186.0)
    aqi_roll_mean_7: float    = Field(..., example=175.0)
    aqi_roll_std_7: float     = Field(..., example=22.0)
    aqi_roll_max_3: float     = Field(..., example=198.0)

    # Weather for tomorrow (forecast)
    temp_c: float             = Field(..., ge=-5, le=50, example=18.0)
    humidity_pct: float       = Field(..., ge=0, le=100, example=72.0)
    wind_speed_ms: float      = Field(..., ge=0, le=20, example=2.1)
    precipitation_mm: float   = Field(default=0.0, ge=0, le=100, example=0.0)
    wind_dir_deg: float       = Field(default=270.0, ge=0, le=360, example=270.0)

    # Context flags
    is_fog_season: int        = Field(..., ge=0, le=1, description="1 if Nov-Feb", example=1)
    is_weekend: int           = Field(default=0, ge=0, le=1, example=0)
    is_ramadan: int           = Field(default=0, ge=0, le=1, example=0)
    month: int                = Field(..., ge=1, le=12, example=11)
    day_of_week: int          = Field(..., ge=0, le=6, description="0=Mon", example=1)


class AQIOutput(BaseModel):
    predicted_aqi: float
    aqi_category: str
    confidence_pct: float
    pm25_estimate: float
    health_advisory: str
    color_code: str
    model_mae: float
    input_features_used: int


# ── Helper: derive full feature vector ───────────────────────────────────────
def build_feature_vector(inp: AQIInput) -> np.ndarray:
    wind_x = inp.wind_speed_ms * math.cos(math.radians(inp.wind_dir_deg))
    wind_y = inp.wind_speed_ms * math.sin(math.radians(inp.wind_dir_deg))
    heat_index = inp.temp_c * (1 + 0.0055 * (inp.humidity_pct - 58))
    inversion_risk = int(inp.temp_c < 15 and inp.humidity_pct > 65)
    dispersion = inp.wind_speed_ms / (inp.humidity_pct / 100 + 0.1)
    month_sin = math.sin(2 * math.pi * inp.month / 12)
    month_cos = math.cos(2 * math.pi * inp.month / 12)
    dow_sin   = math.sin(2 * math.pi * inp.day_of_week / 7)
    dow_cos   = math.cos(2 * math.pi * inp.day_of_week / 7)

    vec = [
        inp.aqi_today,        # aqi_lag_1
        inp.aqi_yesterday,    # aqi_lag_2
        inp.aqi_2days_ago,    # aqi_lag_3
        inp.aqi_6days_ago,    # aqi_lag_6
        inp.aqi_7days_ago,    # aqi_lag_7
        inp.aqi_roll_mean_3,
        inp.aqi_roll_mean_7,
        inp.aqi_roll_std_7,
        inp.aqi_roll_max_3,
        inp.pm25_today,       # pm25_lag_1
        inp.pm25_3days_ago,   # pm25_lag_3
        inp.temp_c,
        inp.humidity_pct,
        inp.wind_speed_ms,
        inp.precipitation_mm,
        wind_x,
        wind_y,
        heat_index,
        inversion_risk,
        dispersion,
        month_sin,
        month_cos,
        dow_sin,
        dow_cos,
        inp.is_fog_season,
        inp.is_weekend,
        inp.is_ramadan,
    ]
    return np.array(vec, dtype=float).reshape(1, -1)


def aqi_to_pm25(aqi: float) -> float:
    if aqi <= 50:    return aqi * 12 / 50
    elif aqi <= 100: return 12 + (aqi-50) * 23.4/50
    elif aqi <= 150: return 35.4 + (aqi-100) * 20/50
    elif aqi <= 200: return 55.4 + (aqi-150) * 54.3/50
    elif aqi <= 300: return 150.4 + (aqi-200) * 99.9/100
    else:            return 250.4 + (aqi-300) * 149.9/200


CATEGORY_INFO = {
    "Good":                              ("#00C853", "Air quality is satisfactory. Outdoor activities are safe for everyone."),
    "Moderate":                          ("#FFD600", "Acceptable air quality. Unusually sensitive people should consider limiting prolonged outdoor exertion."),
    "Unhealthy for Sensitive Groups":    ("#FF6D00", "Sensitive groups (children, elderly, people with respiratory issues) should limit outdoor activity."),
    "Unhealthy":                         ("#DD2C00", "Everyone may begin to experience health effects. Sensitive groups should avoid outdoor activity."),
    "Very Unhealthy":                    ("#6200EA", "Health alert: everyone should avoid prolonged outdoor exertion. Wear N95 mask if going out."),
    "Hazardous":                         ("#37474F", "HEALTH EMERGENCY. Avoid all outdoor activity. Stay indoors with windows closed."),
}


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Lahore AQI Forecasting API",
        "version": "1.0.0",
        "endpoints": ["/predict", "/health", "/model-info"],
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": True}


@app.get("/model-info")
def model_info():
    return META


@app.post("/predict", response_model=AQIOutput)
def predict(inp: AQIInput):
    try:
        X = build_feature_vector(inp)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature error: {e}")

    # Ensemble: average XGBoost and LightGBM regression predictions
    pred_lgb = float(lgb_reg.predict(X)[0])
    pred_xgb = float(xgb_reg.predict(X)[0])
    pred_aqi = round((pred_lgb * 0.4 + pred_xgb * 0.6), 1)  # XGB weighted slightly more
    pred_aqi = max(0, min(600, pred_aqi))

    # Classification probabilities for confidence
    proba = lgb_cls.predict_proba(X)[0]
    cls_idx = int(np.argmax(proba))
    confidence = round(float(proba[cls_idx]) * 100, 1)
    category = le.classes_[cls_idx]

    color, advisory = CATEGORY_INFO.get(category, ("#888888", "No advisory available."))
    pm25 = round(aqi_to_pm25(pred_aqi), 1)

    return AQIOutput(
        predicted_aqi=pred_aqi,
        aqi_category=category,
        confidence_pct=confidence,
        pm25_estimate=pm25,
        health_advisory=advisory,
        color_code=color,
        model_mae=META["metrics"]["xgb_reg_mae"],
        input_features_used=len(FEATURES),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
