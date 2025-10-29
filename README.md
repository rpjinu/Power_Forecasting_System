# Power_Forecasting_System
"Power Forecasting and Anomaly Detection System: predicts future energy production and consumption while identifying downtime anomalies using interpretable ML models."

<src="https://github.com/rpjinu/Power_Forecasting_System/blob/main/project_image.png">

# Multi-Site Operations Forecast & Alerting Pipeline

**One-line:** Power Forecasting and Anomaly Detection System: predicts future energy production and consumption while identifying downtime anomalies using interpretable ML models.

---

## Overview

This repository contains a reproducible pipeline to:

* Clean and merge multi-file daily operations data (`operations_daily_day*.csv`) and site metadata (`site_meta.csv`).
* Perform EDA and feature engineering for per-site daily time-series.
* Forecast the next **14 days** for `units_produced` and `power_kwh` using a baseline and an improved ML model (XGBoost).
* Detect downtime/anomaly events using an interpretable rule-based method and produce `alerts.csv`.
* Provide a minimal FastAPI and an optional Typer CLI to serve forecasts and anomalies.
* Deliver a 1-page executive brief summarizing results and operational triggers.

This README documents step-by-step instructions to reproduce results locally.

---

## Repository structure (recommended)

```
/notebooks/
  01_EDA.ipynb
  02_features_models.ipynb
  03_anomaly.ipynb
/src/
  loader.py
  features.py
  models.py
  anomaly.py
  save_models.py
/app/
  main.py        # FastAPI
  cli.py         # optional Typer CLI
/models/
  pipe_units.joblib
  pipe_power.joblib
/outputs/
  alerts.csv
  forecast_units.csv
  forecast_power.csv
requirements.txt
README.md
executive_brief.pdf
```

---

## Quick start (run locally)

Make sure you have Python **3.10+**. Clone the repo and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate     # Windows PowerShell
pip install -r requirements.txt
```

**Example `requirements.txt`** (minimal):

```
pandas
numpy
scikit-learn
xgboost
prophet
fastapi
uvicorn
pydantic
joblib
matplotlib
seaborn
```

> Note: `prophet` is optional and may require additional system deps; XGBoost is used as the improved model here.

---

## Data

Place your data files (unzipped) under a `data/` folder, or at repository root:

* `operations_daily_day7.csv`, `operations_daily_day14.csv`, ..., `operations_daily_day365.csv` (13 files)
* `site_meta.csv`

**Primary combined file created during preprocessing**: `operations_cleaned.csv` or `operations_combined.csv`.

---

## Reproducible steps (detailed)

Follow these steps in order. Each step corresponds to a notebook or script in `/notebooks/` or `/src/`.

### 1) Combine & Load

* Unzip the archive and extract the 13 `operations_daily_*.csv` files and `site_meta.csv`.
* Combine all daily CSVs into a single DataFrame and remove duplicates.

```python
# src/loader.py (example)
import os
import pandas as pd

def load_ops(folder='data/'):
    files = [f for f in os.listdir(folder) if f.startswith('operations_daily') and f.endswith('.csv')]
    dfs = [pd.read_csv(os.path.join(folder,f)) for f in files]
    ops = pd.concat(dfs, ignore_index=True)
    ops.drop_duplicates(inplace=True)
    ops['date'] = pd.to_datetime(ops['date'])
    return ops

ops = load_ops('data/')
meta = pd.read_csv('data/site_meta.csv')
ops = ops.merge(meta, on='site_id', how='left')
ops.to_csv('operations_cleaned.csv', index=False)
```

### 2) EDA

Open `/notebooks/01_EDA.ipynb`. Perform checks:

* `df.info()`, `df.describe()` and missing values heatmap.
* Time coverage per site, distributions, correlations, seasonal trends, and region comparisons.
* Save important plots for the executive brief.

### 3) Feature Engineering

Open `/notebooks/02_features_models.ipynb` and create:

* Time features: `day_of_week`, `month`, `is_weekend`, `week_of_year`.
* Lag features: `units_lag_1, units_lag_3, units_lag_7`, and `power_lag_*`.
* Rolling features: `units_roll7`, `power_roll7`.
* Normalize features: `units_per_hour = units_produced / shift_hours_per_day`.
* Encode categorical: **label encode** `region` (or `site_id`) — we keep region as label-encoded for XGBoost.

Save final features:

```python
ops.to_csv('operations_features.csv', index=False)
```

### 4) Train/Test Split (time-aware)

* For each site/region, hold out the **last 14 days** as a test set (forecast horizon).
* Training set = all days before the last 14 days.

```python
# simple split
train_df = df.groupby('region').apply(lambda g: g.iloc[:-14]).reset_index(drop=True)
test_df = df.groupby('region').apply(lambda g: g.iloc[-14:]).reset_index(drop=True)
```

### 5) Baseline Model

* Baseline = **7-day rolling mean per site/region** (simple and interpretable).
* Create baseline predictions for the test horizon and compute MAE/MAPE.

### 6) Improved Model — XGBoost Pipeline

* Build an `sklearn.pipeline.Pipeline` that **label-encodes `region`** and trains `XGBRegressor`.
* Train two pipelines: one for `units_produced`, one for `power_kwh`.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from xgboost import XGBRegressor

def label_encode_region(X):
    X = X.copy()
    X['region'] = LabelEncoder().fit_transform(X['region'].astype(str))
    return X

pipe = Pipeline([
    ('label_region', FunctionTransformer(label_encode_region, validate=False)),
    ('model', XGBRegressor(n_estimators=200, max_depth=5, random_state=42))
])
```

* Fit on `X_train` / `y_train`, predict on `X_test`.
* Evaluate MAE & MAPE and save per-region and overall metrics to `outputs/evaluation_mae_mape_by_region.csv`.

### 7) Save models

After training, save pipelines with `joblib`:

```python
import joblib
joblib.dump(pipe_units, 'models/pipe_units.joblib')
joblib.dump(pipe_power, 'models/pipe_power.joblib')
```

### 8) Anomaly Detection (interpretable)

* Algorithm (per site/region):

  1. Compute `units_per_hour`.
  2. Compute lagged rolling median `expected_per_hour = rolling_median(units_per_hour, window=7).shift(1)`.
  3. Flag days where `units_per_hour < THRESH_FRAC * expected_per_hour` (e.g., THRESH_FRAC=0.35).
  4. Group consecutive flagged days and keep events with `duration >= MIN_DAYS` (e.g., 2).
  5. Write `outputs/alerts.csv` with event-level summary fields.

Script: `/src/anomaly.py` (callable from CLI).

### 9) API & CLI

* Minimal FastAPI app (`/app/main.py`) exposes endpoints:

  * `GET /forecast?site=South&start=YYYY-MM-DD&end=YYYY-MM-DD`
  * `GET /anomalies?site=South&start=&end=`
* Load saved models and `outputs/` CSVs on startup. Use `pydantic` models for response validation.

Or use Typer CLI (`/app/cli.py`) for quick queries:

```bash
python app/cli.py forecast South --start 2025-12-18 --end 2025-12-31
python app/cli.py anomalies South
```

### 10) Executive brief

* Create a one-page PDF summarizing:

  * What you built and why
  * Data summary and key EDA findings
  * MAE/MAPE table (baseline vs improved for both targets)
  * Top 3 insights and sample alerts
  * Automation triggers (e.g., thresholds for alerting)

---

## Output files (what to submit)

* `/outputs/alerts.csv` — downtime alerts
* `/outputs/forecast_units.csv` — 14-day units_produced forecasts (baseline & improved)
* `/outputs/forecast_power.csv` — 14-day power_kwh forecasts (baseline & improved)
* `/models/pipe_units.joblib`, `/models/pipe_power.joblib`
* `/executive_brief.pdf`
* Notebooks and code under `/notebooks/` and `/src/`

---

## Tips & Good practices

* Freeze random seed: `random_state=42` across training for reproducibility.
* Document assumptions in the executive brief (thresholds, window sizes).
* Keep the pipeline lightweight (no heavy hyperparameter search) due to time limits.
* Validate that `outputs/*.csv` follow the required format before zipping the repo.

---

## Troubleshooting

* If XGBoost installation fails, try `pip install xgboost` or use `lightgbm` as an alternative.
* If Prophet is required, install system dependencies (e.g., pystan). Prophet is optional.

---

## Contact

gmail- jinupradhan123@gmail.com

Good luck 
