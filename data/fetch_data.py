import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# ─── 1. Fetch real weather data from Open-Meteo (free, no key needed) ──────
print("Fetching weather data from Open-Meteo...")
end_date = datetime(2025, 4, 28)
start_date = end_date - timedelta(days=179)

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 31.5497,
    "longitude": 74.3436,
    "start_date": start_date.strftime("%Y-%m-%d"),
    "end_date": end_date.strftime("%Y-%m-%d"),
    "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation,surface_pressure",
    "timezone": "Asia/Karachi",
    "wind_speed_unit": "ms"
}

resp = requests.get(url, params=params, timeout=30)
weather_raw = resp.json()
print(f"  Status: {resp.status_code}, Hours: {len(weather_raw['hourly']['time'])}")

wdf = pd.DataFrame({
    "datetime": pd.to_datetime(weather_raw["hourly"]["time"]),
    "temp": weather_raw["hourly"]["temperature_2m"],
    "humidity": weather_raw["hourly"]["relative_humidity_2m"],
    "wind_speed": weather_raw["hourly"]["wind_speed_10m"],
    "wind_dir": weather_raw["hourly"]["wind_direction_10m"],
    "precipitation": weather_raw["hourly"]["precipitation"],
    "pressure": weather_raw["hourly"]["surface_pressure"],
})
print(f"  Weather rows: {len(wdf)}")
print(wdf[["datetime","temp","humidity","wind_speed"]].head(3).to_string())

# ─── 2. WAQI current reading (verify station) ───────────────────────────────
print("\nChecking WAQI for Lahore...")
r = requests.get("https://api.waqi.info/feed/lahore/?token=demo", timeout=15)
current = r.json()
print(f"  Status: {current['status']}")
if current['status'] == 'ok':
    d = current['data']
    print(f"  Current AQI: {d['aqi']}")
    print(f"  Station: {d['city']['name']}")
    print(f"  Pollutants: {list(d['iaqi'].keys())}")
    # Save current snapshot
    with open("/home/claude/aqi_project/data/current_aqi.json", "w") as f:
        json.dump(d, f, indent=2, default=str)

# Save weather
wdf.to_csv("/home/claude/aqi_project/data/weather_raw.csv", index=False)
print("\nSaved: weather_raw.csv, current_aqi.json")
