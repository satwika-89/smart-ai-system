# Smart City AI System

Overview

This repository is a starter scaffold for a Smart City AI System that predicts and analyzes air pollution (AQI), with scaffolds for traffic, crowd, and accident models.

Goals

- Build a reproducible data + ML + UI pipeline for air quality forecasting and explainability.
- Provide a Streamlit app with data dashboards, prediction, forecasting, and model management.

Quick start

1. Create and activate a Python virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt
pip install shap statsmodels   # optional: SHAP and statsmodels for explainability and ARIMA
```

2. Prepare data and features:

```bash
python -m src.data.aqi_loader
python -m src.preprocessing.airquality_pipeline
```

3. Train advanced models (optional):

```bash
python -m src.models.train_pollution_advanced --target "C6H6(GT)"
```

4. Run the Streamlit dashboard:

```bash
streamlit run app/streamlit_app.py
```

Included pages

- Home — project overview and system flow
- Data Dashboard — summary stats, missing values, time-series plots
- Prediction — manual/CSV prediction and retrain
- Forecasting — short-term forecasts (ARIMA if available)
- Explainability — SHAP feature importance (if generated)
- Model & Training — metrics, model versions, retrain
- Upload & Insights — quick analyses for uploaded CSVs

Artifacts & versioning

- Cleaned datasets: `src/data/processed/pollution_v{n}_clean.csv`
- Feature files: `src/data/processed/airquality_features_v{n}.csv`
- Models & metrics: `src/models/artifacts/model_v{n}.joblib`, `metrics_v{n}.json`

Notes

- SHAP and statsmodels are optional but recommended for explainability and forecasting.
- The project aims to showcase data engineering, modeling, explainability, and an integrated UI useful for decision-making.

License

Add a license if desired.

