"""
Lahore AQI Dataset Generator
Based on real published data:
- IQAir World Air Quality Report 2022/2023: Lahore avg PM2.5 ~97 µg/m³
- Pakistan EPA seasonal patterns: worst Oct-Feb (fog/smog season)
- Typical Lahore AQI range: 50-500+, peaks 300-500 in winter
- Open-Meteo climatology for Lahore (31.55°N, 74.34°E)
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

np.random.seed(42)

# ── Date range: 365 days ending April 28 2025 ──────────────────────────────
end_dt = datetime(2025, 4, 28)
start_dt = end_dt - timedelta(days=364)
dates = [start_dt + timedelta(days=i) for i in range(365)]

# ── Seasonal temperature profile for Lahore (°C) ────────────────────────────
# Jan:12, Feb:15, Mar:21, Apr:28, May:34, Jun:37, Jul:34, Aug:33, Sep:30, Oct:25, Nov:18, Dec:13
monthly_temp = {1:12,2:15,3:21,4:28,5:34,6:37,7:34,8:33,9:30,10:25,11:18,12:13}

# ── AQI baseline by month (real Lahore pattern) ─────────────────────────────
# Source: IQAir, Pak-EPA monitoring data
# Winter smog season: Oct-Feb peaks 200-400+
# Summer: 80-150 (heat disperses, but dust storms)
monthly_aqi_mean = {1:210, 2:190, 3:140, 4:110, 5:120, 6:130,
                    7:105, 8:100, 9:115, 10:160, 11:220, 12:240}
monthly_aqi_std  = {1:70,  2:65,  3:45,  4:35,  5:45,  6:55,
                    7:35,  8:30,  9:40,  10:55, 11:70,  12:80}

records = []

for d in dates:
    m = d.month
    doy = d.timetuple().tm_yday

    # Temperature with daily noise
    base_temp = monthly_temp[m]
    temp = base_temp + np.random.normal(0, 2.5)

    # Humidity (%) — higher in monsoon Jul-Sep, lower in spring
    hum_base = {1:65,2:55,3:45,4:40,5:38,6:42,7:72,8:75,9:65,10:55,11:65,12:70}
    humidity = np.clip(hum_base[m] + np.random.normal(0, 8), 20, 98)

    # Wind speed (m/s) — lower in winter inversions
    wind_base = {1:1.8,2:2.2,3:2.8,4:3.2,5:3.5,6:3.8,7:3.5,8:3.2,9:2.8,10:2.3,11:1.9,12:1.7}
    wind_speed = np.clip(wind_base[m] + np.random.normal(0, 0.7), 0.1, 9.0)

    # Wind direction (degrees) — westerlies in winter, SW monsoon Jul-Sep
    wind_dir = np.random.uniform(200, 280) if m in [7,8,9] else np.random.uniform(270, 360)

    # Precipitation (mm)
    precip_prob = {1:0.08,2:0.10,3:0.12,4:0.10,5:0.05,6:0.08,
                   7:0.35,8:0.40,9:0.25,10:0.08,11:0.05,12:0.07}
    precipitation = np.random.exponential(3.5) if np.random.random() < precip_prob[m] else 0.0

    # Fog season binary (Nov-Feb)
    is_fog_season = 1 if m in [11,12,1,2] else 0

    # Ramadan 2024/2025 (approx March 11 - April 9 2024, March 1 - March 30 2025)
    is_ramadan = 1 if (d >= datetime(2024,3,11) and d <= datetime(2024,4,9)) or \
                      (d >= datetime(2025,3,1) and d <= datetime(2025,3,30)) else 0

    # Weekend
    is_weekend = 1 if d.weekday() >= 5 else 0

    # Day of week, month features
    dow = d.weekday()  # 0=Mon

    # ── AQI calculation (realistic causal model) ─────────────────────────────
    aqi_base = monthly_aqi_mean[m]

    # Wind disperses pollutants: strong inverse relationship
    wind_effect = -18 * (wind_speed - wind_base[m])

    # Temperature inversion in cold weather amplifies AQI
    inversion_effect = 30 if (temp < 15 and humidity > 60) else 0

    # Rain washes pollutants: strong negative effect
    rain_effect = -60 * min(precipitation / 5, 1.0)

    # Weekend: slightly lower industrial + traffic emissions
    weekend_effect = -12 if is_weekend else 0

    # Ramadan: night activity shifts, slightly different traffic
    ramadan_effect = -8 if is_ramadan else 0

    # Dust storm effect (hot dry days May-Jun)
    dust_storm = 0
    if m in [5,6] and temp > 36 and humidity < 35 and np.random.random() < 0.07:
        dust_storm = np.random.uniform(80, 200)

    aqi_raw = (aqi_base + wind_effect + inversion_effect + rain_effect
               + weekend_effect + ramadan_effect + dust_storm
               + np.random.normal(0, monthly_aqi_std[m]))

    aqi = float(np.clip(aqi_raw, 20, 550))

    # PM2.5 (µg/m³) — directly correlated with AQI for this range
    # AQI 0-50: PM2.5 0-12, 51-100: 12-35.4, 101-150: 35.4-55.4 etc.
    if aqi <= 50:    pm25 = aqi * 12 / 50
    elif aqi <= 100: pm25 = 12 + (aqi-50) * 23.4/50
    elif aqi <= 150: pm25 = 35.4 + (aqi-100) * 20/50
    elif aqi <= 200: pm25 = 55.4 + (aqi-150) * 54.3/50
    elif aqi <= 300: pm25 = 150.4 + (aqi-200) * 99.9/100
    else:            pm25 = 250.4 + (aqi-300) * 149.9/200
    pm25 = float(np.clip(pm25 + np.random.normal(0, 2), 0, 500))

    # AQI category (US EPA standard)
    if aqi <= 50:    category = "Good"
    elif aqi <= 100: category = "Moderate"
    elif aqi <= 150: category = "Unhealthy for Sensitive Groups"
    elif aqi <= 200: category = "Unhealthy"
    elif aqi <= 300: category = "Very Unhealthy"
    else:            category = "Hazardous"

    records.append({
        "date": d.strftime("%Y-%m-%d"),
        "month": m,
        "day_of_week": dow,
        "is_weekend": is_weekend,
        "is_fog_season": is_fog_season,
        "is_ramadan": is_ramadan,
        "temp_c": round(temp, 1),
        "humidity_pct": round(humidity, 1),
        "wind_speed_ms": round(wind_speed, 2),
        "wind_dir_deg": round(wind_dir, 1),
        "precipitation_mm": round(precipitation, 2),
        "aqi": round(aqi, 1),
        "pm25_ugm3": round(pm25, 1),
        "aqi_category": category,
    })

df = pd.DataFrame(records)
output_path = Path(__file__).resolve().parent / "lahore_aqi_dataset.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print(f"Dataset generated: {len(df)} rows")
print(f"Date range: {df['date'].min()} → {df['date'].max()}")
print(f"\nAQI statistics:")
print(df["aqi"].describe().round(1).to_string())
print(f"\nAQI category distribution:")
print(df["aqi_category"].value_counts().to_string())
print(f"\nMonthly average AQI:")
print(df.groupby("month")["aqi"].mean().round(1).to_string())
