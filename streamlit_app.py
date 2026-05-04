"""Minimal Streamlit dashboard to interact with the Smart City pipeline."""
import sys
from pathlib import Path
import streamlit as st

# Ensure project root is on sys.path so `import src` works when Streamlit
# launches from a different working directory.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.integration.pipeline import SmartCityPipeline
import pandas as pd
import joblib
import subprocess
import threading
import time
import re
from io import StringIO
import json
import logging
from src.models.forecast import forecast_series
from src.utils.logger import logger

ARTIFACTS_DIR = ROOT / "src" / "models" / "artifacts"


def _scan_models():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    models = []
    for p in ARTIFACTS_DIR.glob("model_v*.joblib"):
        name = p.stem
        # try to find matching metrics json
        metrics = None
        metrics_path = ARTIFACTS_DIR / (f"metrics_{name.split('_v')[-1]}.json")
        # fallback: any metrics_v*.json
        if metrics_path.exists():
            try:
                import json

                with open(metrics_path, "r", encoding="utf8") as fh:
                    metrics = json.load(fh)
            except Exception:
                metrics = None
        models.append({"model_path": p, "name": name, "metrics": metrics})
    # sort by filename
    models = sorted(models, key=lambda x: x["name"])
    return models


@st.cache_resource
def list_available_models():
    return _scan_models()


@st.cache_resource
def load_model_from_path(path: str):
    try:
        return joblib.load(path)
    except Exception:
        return None


def load_scaler_for_model_version(version: str):
    # common scaler naming
    candidates = list(ARTIFACTS_DIR.glob(f"scaler_v{version}.joblib"))
    if not candidates:
        candidates = list(ARTIFACTS_DIR.glob("scaler_*.joblib"))
    if candidates:
        try:
            return joblib.load(candidates[-1])
        except Exception:
            return None
    return None

st.set_page_config(page_title="Smart City AI", layout="wide")

st.title("Smart City AI — Dashboard (Starter)")

pipeline = SmartCityPipeline()

if st.sidebar.button("Load models"):
    pipeline.load_models()
    st.sidebar.success("Models loaded (placeholders)")

page = st.sidebar.selectbox("Page", ["Home", "Data Dashboard", "Prediction", "Forecasting", "Explainability", "Model & Training", "Upload & Insights"]) 

st.header(page)

def align_features(df: pd.DataFrame, model, scaler):
    # Determine required features from the trained model if available
    if model is None:
        st.error("No model loaded — train or load a model first.")
        return None

    if hasattr(model, "feature_names_in_"):
        required = list(model.feature_names_in_)
    else:
        required = list(df.columns)

    # pick columns from df, fill missing with zeros
    X = pd.DataFrame(index=df.index)
    for col in required:
        if col in df.columns:
            X[col] = df[col]
        else:
            X[col] = 0.0

    # apply scaler if present
    if scaler is not None:
        try:
            X_values = scaler.transform(X.values)
            X = pd.DataFrame(X_values, columns=X.columns, index=X.index)
        except Exception:
            pass

    return X


def predict_df(df: pd.DataFrame):
    models = list_available_models()
    if not models:
        st.warning("Model not found. Run training first (see 'Retrain model').")
        return None
    # use latest model by name ordering
    sel = models[-1]
    model = load_model_from_path(str(sel["model_path"]))
    version_match = re.search(r"_v(\d+)", sel["name"])
    version = version_match.group(1) if version_match else None
    scaler = load_scaler_for_model_version(version)

    if model is None:
        st.warning("Failed to load the selected model.")
        return None

    X = align_features(df, model, scaler)
    if X is None:
        return None

    try:
        preds = model.predict(X)
    except Exception as exc:
        st.error(f"Prediction error: {exc}")
        return None

    return preds
if page == "Home":
    st.subheader("Smart City AI — Overview")
    st.markdown("""
    **Goal:** Predict and explain air pollution (AQI) and provide forecasting and visualizations for decision-making.

    **System layers:**
    - Data & Pipeline: loader, preprocessing, versioned datasets
    - ML: training, comparison, explainability
    - Application: Streamlit UI for prediction, forecasting, and insights
    """)
    st.markdown("**Flow:** Raw Data → Cleaning → Feature Engineering → Model Training → Prediction → Visualization")

elif page == "Data Dashboard":
    st.subheader("Data Dashboard — Cleaned & Processed Data")
    proc_dir = ROOT / "src" / "data" / "processed"
    feat_files = sorted(list(proc_dir.glob("airquality_features_v*.csv")))
    clean_files = sorted(list(proc_dir.glob("pollution_v*_clean.csv")))
    if feat_files:
        df = pd.read_csv(feat_files[-1], index_col=0, parse_dates=True)
        st.write(f"Loaded features: {feat_files[-1].name}")
    elif clean_files:
        df = pd.read_csv(clean_files[-1], index_col=0, parse_dates=True)
        st.write(f"Loaded cleaned data: {clean_files[-1].name}")
    else:
        st.info("No processed data found. Run preprocessing to generate features.")
        df = None

    if df is not None:
        st.markdown("**Summary statistics**")
        st.dataframe(df.describe().T)
        st.markdown("**Missing values (counts)**")
        mv = df.isna().sum()
        st.dataframe(mv[mv>0])
        st.markdown("**Time-series preview**")
        cols = st.multiselect("Plot columns", options=list(df.columns), default=[c for c in df.columns if "C6H6" in c or "CO(GT)" in c][:2])
        if cols:
            st.line_chart(df[cols])

elif page == "Prediction":
    st.subheader("Pollution — Predict target using trained model")

    # Show available models and let user pick a model version
    models = list_available_models()
    model_names = [m["name"] for m in models] if models else []
    sel_model = None
    sel_scaler = None
    if model_names:
        sel_name = st.selectbox("Select model version", options=["(none)"] + model_names, index=0)
        if sel_name and sel_name != "(none)":
            sel = next((m for m in models if m["name"] == sel_name), None)
            if sel:
                sel_model = load_model_from_path(str(sel["model_path"]))
                # try to extract version suffix
                m = re.search(r"_v(\d+)", sel["name"])
                version = m.group(1) if m else None
                sel_scaler = load_scaler_for_model_version(version) if version else None
                st.write("Loaded:", sel["name"])
                if sel["metrics"]:
                    st.metric("Latest metric (example)", sel["metrics"].get("rmse", "-"))
                else:
                    st.info("No metrics metadata found for this model.")
    else:
        st.info("No trained models found in artifacts. Run advanced training.")

    input_mode = st.radio("Input mode", ["Use processed features file", "Upload CSV", "Manual input"])

    df_input = None
    expected_cols = None
    # try to infer expected features from model if available
    if sel_model is not None and hasattr(sel_model, "feature_names_in_"):
        expected_cols = list(sel_model.feature_names_in_)

    if input_mode == "Use processed features file":
        proc_dir = ROOT / "src" / "data" / "processed"
        feat_files = sorted(list(proc_dir.glob("airquality_features_v*.csv")))
        if feat_files:
            proc_path = feat_files[-1]
            df_all = pd.read_csv(proc_path, index_col=0, parse_dates=True)
            st.write("Loaded processed features from:", proc_path.name)
            st.dataframe(df_all.head())
            pick = st.selectbox("Select row by index (datetime)", options=df_all.index.tolist())
            if pick:
                df_input = df_all.loc[[pick]]
        else:
            st.info("Processed features file not found. Run preprocessing to generate it.")

    elif input_mode == "Upload CSV":
        up = st.file_uploader("Upload CSV with feature columns (header required)", type=["csv"])
        if up is not None:
            # improved UX: try index parse, otherwise plain read
            try:
                df_up = pd.read_csv(up, index_col=0, parse_dates=True)
            except Exception:
                up.seek(0)
                df_up = pd.read_csv(up)

            st.write("Preview uploaded file:")
            st.dataframe(df_up.head())

            # auto-map common column variants to expected names if available
            if expected_cols:
                def normalize(s):
                    return re.sub(r"[^a-z0-9]", "", str(s).lower())

                norm_to_col = {normalize(c): c for c in df_up.columns}
                mapped = {}
                for exp in expected_cols:
                    key = normalize(exp)
                    if key in norm_to_col:
                        mapped[exp] = norm_to_col[key]
                if mapped:
                    st.write("Auto-mapped columns:")
                    st.json(mapped)
                    # create a aligned df for prediction columns
                    df_mapped = pd.DataFrame(index=df_up.index)
                    for exp in expected_cols:
                        src = mapped.get(exp)
                        if src is not None:
                            df_mapped[exp] = df_up[src]
                        else:
                            df_mapped[exp] = 0.0
                    df_up_aligned = df_mapped
                else:
                    df_up_aligned = df_up
            else:
                df_up_aligned = df_up

            # let user pick rows to predict
            rows = st.multiselect("Select rows to predict (by index)", options=list(df_up_aligned.index.astype(str)), default=[str(df_up_aligned.index[0])])
            if rows:
                df_input = df_up_aligned.loc[[pd.to_datetime(r) for r in rows]]

    else:  # Manual input
        st.markdown("Provide basic pollutant and time inputs. Missing features are filled with zeros.")
        col1, col2, col3 = st.columns(3)
        date = col1.date_input("Date")
        hour = col2.slider("Hour", 0, 23, 12)
        co = col3.number_input("CO(GT)", value=1.0)
        no2 = st.number_input("NO2(GT)", value=20.0)
        temp = st.number_input("T (deg C)", value=20.0)
        rh = st.number_input("RH (%)", value=50.0)

        # Build minimal DataFrame
        dt = pd.to_datetime(str(date) + f" {hour:02d}:00:00")
        df_input = pd.DataFrame([{"CO(GT)": co, "NO2(GT)": no2, "T": temp, "RH": rh}], index=[dt])

    if df_input is not None:
        st.write("Input for prediction:")
        st.dataframe(df_input)

        def _predict_with_selected(df_local):
            if sel_model is None:
                st.warning("No model selected. Choose a model version to predict.")
                return None
            X = align_features(df_local, sel_model, sel_scaler)
            if X is None:
                return None
            try:
                preds = sel_model.predict(X)
            except Exception as exc:
                st.error(f"Prediction error: {exc}")
                return None
            return preds

        if st.button("Predict"):
            preds = _predict_with_selected(df_input)
            if preds is not None:
                if len(preds) == 1:
                    st.metric("Prediction", float(preds[0]))
                else:
                    st.write(pd.DataFrame({"prediction": preds}, index=df_input.index))

    st.markdown("---")

    # Background retrain support: start training in background thread
    def _run_training_bg(advanced: bool = True):
        cmd = ["python", "-m", "src.models.train_pollution_advanced"] if advanced else ["python", "-m", "src.models.train_pollution"]
        try:
            p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            # store PID or process in session_state
            st.session_state["training_pid"] = p.pid
            # optionally detach and return
            return p
        except Exception as e:
            st.error(f"Failed to start training: {e}")
            return None

    if st.button("Retrain model (advanced)"):
        st.info("Retraining model started in background. Check logs in artifacts folder when done.")
        _run_training_bg(advanced=True)

    # Model comparison / SHAP: look for metrics files
    st.subheader("Model Comparison")
    artifacts = list((ROOT / "src" / "models" / "artifacts").glob("metrics_v*.json"))
    if artifacts:
        # show latest
        latest = sorted(artifacts, key=lambda p: p.name)[-1]
        try:
            mj = pd.read_json(latest)
            st.write(f"Loaded metrics: {latest.name}")
            st.dataframe(mj)
        except Exception:
            with open(latest, "r", encoding="utf8") as fh:
                data = json.load(fh)
            st.table(pd.DataFrame(data))

        # show SHAP if available
        shap_files = list((ROOT / "src" / "models" / "artifacts").glob("shap_importance_v*.csv"))
        if shap_files:
            shap_latest = sorted(shap_files, key=lambda p: p.name)[-1]
            try:
                shap_df = pd.read_csv(shap_latest)
                st.subheader("Feature importance (SHAP)")
                st.bar_chart(shap_df.set_index("feature").head(20))
            except Exception as e:
                st.warning(f"Failed to load SHAP file: {e}")
    else:
        st.info("No metrics found. Run 'Retrain model' to generate model comparison artifacts.")

elif page == "Forecasting":
    st.subheader("Forecasting — short-term pollution forecasts")
    proc_dir = ROOT / "src" / "data" / "processed"
    feat_files = sorted(list(proc_dir.glob("airquality_features_v*.csv")))
    if not feat_files:
        st.info("No features found. Run preprocessing first.")
    else:
        df = pd.read_csv(feat_files[-1], index_col=0, parse_dates=True)
        target_cols = [c for c in df.columns if df[c].dtype.kind in "fi"]
        target = st.selectbox("Target series (numeric)", options=target_cols, index=0)
        steps = st.selectbox("Forecast horizon", [24, 72, 168])
        series = df[target].dropna()
        if len(series) < 10:
            st.warning("Not enough history to forecast")
        else:
            preds = forecast_series(series, steps=steps)
            st.line_chart(pd.concat([series.tail(200), preds]))
            if preds.max() > 150:
                st.error("High pollution risk forecasted in the horizon!")

elif page == "Explainability":
    st.subheader("Explainability (SHAP)")
    art_dir = ROOT / "src" / "models" / "artifacts"
    shap_files = sorted(list(art_dir.glob("shap_importance_v*.csv")))
    if shap_files:
        shap_df = pd.read_csv(shap_files[-1])
        st.write(f"Loaded SHAP importance: {shap_files[-1].name}")
        st.dataframe(shap_df.head(50))
        st.bar_chart(shap_df.set_index("feature").head(20))
    else:
        st.info("No SHAP artifacts found. Run advanced training with SHAP available (install shap).")

elif page == "Model & Training":
    st.subheader("Model & Training — metrics, versions, retrain")
    art_dir = ROOT / "src" / "models" / "artifacts"
    metric_files = sorted(list(art_dir.glob("metrics_v*.json")))
    model_files = sorted(list(art_dir.glob("model_v*.joblib")))
    st.write("Metrics files:")
    if metric_files:
        for mf in metric_files:
            st.write(mf.name)
        # show latest
        with open(metric_files[-1], "r", encoding="utf8") as fh:
            metrics = json.load(fh)
        st.table(pd.DataFrame(metrics))
    else:
        st.info("No metrics found. Run advanced training to generate metrics.")

    st.write("Models:")
    for m in model_files:
        st.write(m.name)

    if st.button("Retrain (advanced)"):
        st.info("Retraining (advanced)...")
        with st.spinner("Running advanced training..."):
            try:
                proc = subprocess.run(["python", "-m", "src.models.train_pollution_advanced"], capture_output=True, text=True, cwd=str(ROOT))
                out = proc.stdout + "\n" + proc.stderr
                st.text_area("Advanced training output", out, height=400)
                logger.info("Advanced retrain invoked via Streamlit")
            except Exception as e:
                st.error(f"Retrain failed: {e}")

elif page == "Upload & Insights":
    st.subheader("Upload CSV & Quick Insights")
    up = st.file_uploader("Upload CSV (index col optional)", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up, index_col=0, parse_dates=True)
        except Exception:
            df = pd.read_csv(up)
        st.markdown("**Preview**")
        st.dataframe(df.head())
        st.markdown("**Summary**")
        st.dataframe(df.describe().T)
        st.markdown("**Missing values**")
        mv = df.isna().sum()
        st.dataframe(mv[mv>0])
        numeric = df.select_dtypes(include=["number"]).columns.tolist()
        if numeric:
            st.line_chart(df[numeric].iloc[:200])
        if st.button("Run predictions on uploaded data"):
            # attempt to align and predict
            try:
                preds = predict_df(df)
                if preds is not None:
                    st.write(pd.DataFrame({"prediction": preds}, index=df.index))
            except Exception as e:
                st.error(f"Prediction failed: {e}")

else:
    st.markdown("This is a starter dashboard. Use the sidebar to navigate pages.")
