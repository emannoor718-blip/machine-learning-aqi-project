# Lahore AQI Next-Day Forecasting

**End-to-end ML project** — predicts tomorrow's Air Quality Index for Lahore, Pakistan.

---

## Project Structure

```
aqi_project/
├── data/
│   ├── generate_dataset.py        # Generate/refresh the dataset
│   ├── fetch_data.py              # Live API fetching (WAQI + Open-Meteo)
│   └── lahore_aqi_dataset.csv     # 365-day dataset
├── models/
│   ├── lgbm_regressor.pkl
│   ├── lgbm_classifier.pkl
│   ├── xgb_regressor.pkl
│   ├── label_encoder.pkl
│   └── model_meta.json            # Metrics + feature list
├── plots/
│   ├── 01_aqi_timeseries.png
│   ├── 02_monthly_aqi.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_shap_importance.png
│   ├── 05_predicted_vs_actual.png
│   ├── 06_confusion_matrix.png
│   └── 07_aqi_distribution.png
├── api/
│   └── main.py                    # FastAPI inference server
├── frontend/
│   └── app.py                     # Streamlit dashboard
├── train_model.py                 # Full training pipeline
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies
```bash
uv add -r requirements.txt
```

### 2. Generate dataset
```bash
python data/generate_dataset.py
```

### 3. Train models
```bash
python train_model.py
```

### 4. Run the API
```bash
uvicorn api.main:app --reload --port 8000
# Docs at: http://localhost:8000/docs
```

### 5. Run the dashboard
```bash
streamlit run frontend/app.py
```

---

## API Reference

### POST /predict

**Request body:**
```json
{
  "aqi_today": 185.0,
  "aqi_yesterday": 195.0,
  "aqi_2days_ago": 178.0,
  "aqi_6days_ago": 160.0,
  "aqi_7days_ago": 155.0,
  "pm25_today": 95.0,
  "pm25_3days_ago": 88.0,
  "aqi_roll_mean_3": 186.0,
  "aqi_roll_mean_7": 175.0,
  "aqi_roll_std_7": 22.0,
  "aqi_roll_max_3": 198.0,
  "temp_c": 14.5,
  "humidity_pct": 78.0,
  "wind_speed_ms": 1.8,
  "precipitation_mm": 0.0,
  "wind_dir_deg": 290.0,
  "is_fog_season": 1,
  "is_weekend": 0,
  "is_ramadan": 0,
  "month": 11,
  "day_of_week": 1
}
```

**Response:**
```json
{
  "predicted_aqi": 201.4,
  "aqi_category": "Very Unhealthy",
  "confidence_pct": 76.5,
  "pm25_estimate": 104.2,
  "health_advisory": "Health alert! Avoid prolonged outdoor exertion...",
  "color_code": "#6200EA",
  "model_mae": 49.0,
  "input_features_used": 27
}
```

---

## Model Performance

| Model | Task | MAE | RMSE |
|-------|------|-----|------|
| XGBoost | Regression | 49.0 | 62.7 |
| LightGBM | Regression | 55.7 | 67.5 |
| LightGBM | Classification | — | F1=0.31 |
| **Ensemble** | **Final** | **~47** | — |

> MAE of 49 on an AQI scale of 0-600 with an average test AQI of ~125.
> Classification F1 is moderate (0.31) due to class imbalance — "Hazardous" days are rare.

---

## Feature Engineering (27 features)

| Group | Features |
|-------|----------|
| AQI lags | aqi_lag_1/2/3/6/7 |
| Rolling stats | roll_mean_3/7, roll_std_7, roll_max_3 |
| PM2.5 lags | pm25_lag_1/3 |
| Weather | temp_c, humidity, wind_speed, precipitation |
| Interactions | wind_x/y, heat_index, inversion_risk, dispersion_capacity |
| Cyclical time | month_sin/cos, dow_sin/cos |
| Context flags | is_fog_season, is_weekend, is_ramadan |

---

## Deploy to Production

### API (Render.com — free tier)
1. Push project to GitHub
2. Create new Web Service on Render
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

### Frontend (Streamlit Cloud — free)
1. Push to GitHub
2. Go to share.streamlit.io
3. Point to `frontend/app.py`
4. Done — public URL in minutes

---

## Live API Integration (with real data)

To use live AQI from WAQI API:
```python
import requests
TOKEN = "your_token_from_waqi.info"
r = requests.get(f"https://api.waqi.info/feed/lahore/?token={TOKEN}")
current_aqi = r.json()["data"]["aqi"]
pm25 = r.json()["data"]["iaqi"]["pm25"]["v"]
```

Free tokens at: https://aqicn.org/api/

For weather: https://api.open-meteo.com (no key needed)

---

## Data Sources

- **IQAir World Air Quality Report 2023** — Lahore baseline statistics
- **WAQI (World Air Quality Index)** — Real-time sensor data
- **Open-Meteo** — Historical and forecast weather data
- **Pakistan EPA** — Seasonal pattern calibration
