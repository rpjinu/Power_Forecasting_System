import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from datetime import timedelta

app = FastAPI(title="Forecast & Alerts API")

# Paths (adjust if needed)
MODELS_DIR = "models"
DATA_PATH = "data/operations_cleaned.csv"
OUTPUT_FORECAST_UNITS = "outputs/forecast_units.csv"
OUTPUT_FORECAST_POWER = "outputs/forecast_power.csv"
ALERTS_PATH = "outputs/alerts.csv"

# Load models (these must exist)
try:
    model_units = joblib.load(f"C:\Users\Ranjan kumar pradhan\.vscode\xgb_units_model.pkl")
    model_power = joblib.load(f"C:\Users\Ranjan kumar pradhan\.vscode\xgb_power_model.pkl")
except Exception as e:
    # Models might be missing at development time; raise a clear message
    print("Warning: Could not load models from models/ -- check files. Error:", e)
    model_units = None
    model_power = None

# Quick helper: read CSV if exists
def _read_csv_if_exists(path, parse_dates=None):
    try:
        df = pd.read_csv(path, parse_dates=parse_dates)
        return df
    except Exception:
        return None

# Load cleaned data (required for feature scaffolding/prediction)
df_all = _read_csv_if_exists(DATA_PATH, parse_dates=["date"])
if df_all is None:
    print(f"Warning: {DATA_PATH} not found. API predict endpoints that rely on data will fail.")

# Endpoint: return precomputed forecasts if exist, otherwise trigger model prediction
@app.get("/forecast")
def get_forecast(site: str = Query(..., description="region or site identifier"),
                 target: str = Query("units_produced", regex="^(units_produced|power_kwh)$"),
                 start: str = Query(None), end: str = Query(None),
                 force_predict: bool = Query(False, description="If true, use models to predict rather than precomputed CSV.")):
    """
    Returns forecast rows for the requested site (region) and date range.
    If precomputed output CSVs are present, they are returned. Otherwise if force_predict==True
    and models + data are available, the API will generate next-14-day forecast using last-known features.
    """
    # Try to read precomputed CSV
    if target == "units_produced":
        fc_df = _read_csv_if_exists(OUTPUT_FORECAST_UNITS, parse_dates=["date"])
    else:
        fc_df = _read_csv_if_exists(OUTPUT_FORECAST_POWER, parse_dates=["date"])

    # If CSV exists and not forcing model generation, return filtered CSV
    if fc_df is not None and not force_predict:
        df_f = fc_df[fc_df["region"] == site]
        if start:
            df_f = df_f[df_f["date"] >= pd.to_datetime(start)]
        if end:
            df_f = df_f[df_f["date"] <= pd.to_datetime(end)]
        if df_f.empty:
            raise HTTPException(status_code=404, detail="No forecast rows found for params.")
        # return as dict list
        return df_f.sort_values("date").to_dict(orient="records")

    # Else attempt to generate forecast using models + last-known features
    if model_units is None or model_power is None or df_all is None:
        raise HTTPException(status_code=500, detail="Model or data missing. Ensure models in /models and data/operations_cleaned.csv exist.")

    # Use last known features for this site/region
    site_rows = df_all[df_all["region"] == site].sort_values("date")
    if site_rows.empty:
        raise HTTPException(status_code=404, detail="Site/region not found in data.")

    last_row = site_rows.iloc[-1].copy()
    # Build horizon dates
    horizon = 14
    future_dates = [last_row["date"] + timedelta(days=i) for i in range(1, horizon+1)]

    # Select features columns used by the model pipeline: derive from model input by using last_row keys minus targets/date
    # Here we assume the model was trained on the same columns in operations_cleaned minus ['date','units_produced','power_kwh']
    feature_cols = [c for c in df_all.columns if c not in ("date","units_produced","power_kwh")]
    # Build DataFrame of size horizon repeating last_row for numeric columns and updating date-driven features
    rows = []
    for dt in future_dates:
        r = last_row[feature_cols].copy()
        # adjust date-derived fields if present
        if "month" in r.index:
            r["month"] = pd.Timestamp(dt).month
        if "day_of_week" in r.index:
            r["day_of_week"] = pd.Timestamp(dt).dayofweek
        if "is_weekend" in r.index:
            r["is_weekend"] = int(pd.Timestamp(dt).dayofweek in (5,6))
        # NOTE: external regressors like temperature/rainfall/holiday_flag are left as last-known values
        r["date"] = dt
        rows.append(r)

    X_future = pd.DataFrame(rows).reset_index(drop=True)
    # Reorder columns to model expected (pipeline will handle label encoding)
    # Predict using appropriate model
    if target == "units_produced":
        preds = model_units.predict(X_future[feature_cols])
        result_df = pd.DataFrame({"region": site, "date": future_dates, "target": target, "forecast": preds})
    else:
        preds = model_power.predict(X_future[feature_cols])
        result_df = pd.DataFrame({"region": site, "date": future_dates, "target": target, "forecast": preds})

    # filter by start/end if provided
    if start:
        result_df = result_df[result_df["date"] >= pd.to_datetime(start)]
    if end:
        result_df = result_df[result_df["date"] <= pd.to_datetime(end)]
    return result_df.sort_values("date").to_dict(orient="records")


# Endpoint: anomalies
@app.get("/anomalies")
def get_anomalies(site: str = Query(...), start: str = Query(None), end: str = Query(None)):
    """
    Returns alert events for a site/region. Reads outputs/alerts.csv.
    """
    alerts_df = _read_csv_if_exists(ALERTS_PATH, parse_dates=["start_date","end_date"])
    if alerts_df is None:
        raise HTTPException(status_code=500, detail="alerts.csv not found in outputs/ - run anomaly script first.")
    df = alerts_df[alerts_df["site"].astype(str) == site]
    if start:
        df = df[pd.to_datetime(df["start_date"]) >= pd.to_datetime(start)]
    if end:
        df = df[pd.to_datetime(df["end_date"]) <= pd.to_datetime(end)]
    if df.empty:
        raise HTTPException(status_code=404, detail="No alerts found for parameters.")
    return df.sort_values("start_date").to_dict(orient="records")