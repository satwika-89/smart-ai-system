
"""Preprocessing pipeline for Air Quality data.

Provides feature engineering utilities (time features, rolling stats,
lags) and a `prepare_for_model` helper that returns X, y and an optional
scaler for normalization.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np

from src.preprocessing.utils import handle_missing_values, normalize_df, split_features_target
from src.data.aqi_loader import load_airquality_uci


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        # try to find a datetime column
        if "datetime" in df.columns:
            df = df.set_index(pd.to_datetime(df["datetime"], errors="coerce"))
        else:
            raise ValueError("DataFrame must have a DatetimeIndex or a 'datetime' column")
    return df.sort_index()


def generate_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_datetime_index(df)
    ts = df.index
    df["hour"] = ts.hour
    df["dayofweek"] = ts.dayofweek
    df["month"] = ts.month
    df["day"] = ts.day
    df["is_weekend"] = ts.dayofweek >= 5
    return df


def add_rolling_features(df: pd.DataFrame, cols: Optional[List[str]] = None, windows: List[int] = [3, 24]) -> pd.DataFrame:
    df = ensure_datetime_index(df)
    num_cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()
    for w in windows:
        rolled = df[num_cols].rolling(window=w, min_periods=1)
        df[[f"{c}_rmean_{w}" for c in num_cols]] = rolled.mean()
        df[[f"{c}_rstd_{w}" for c in num_cols]] = rolled.std().fillna(0)
    return df


def add_lag_features(df: pd.DataFrame, cols: Optional[List[str]] = None, lags: List[int] = [1, 24]) -> pd.DataFrame:
    df = ensure_datetime_index(df)
    num_cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()
    for lag in lags:
        for c in num_cols:
            df[f"{c}_lag_{lag}"] = df[c].shift(lag)
    return df


def prepare_for_model(
    df: pd.DataFrame,
    target_column: str,
    fill_strategy: str = "mean",
    normalize: bool = True,
    dropna_target: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, Optional[object]]:
    df = df.copy()
    df = ensure_datetime_index(df)

    # basic time features and rolling/lags
    df = generate_time_features(df)
    df = add_rolling_features(df)
    df = add_lag_features(df)

    # handle missing values
    df = handle_missing_values(df, strategy=fill_strategy)

    # drop rows with missing target
    if dropna_target:
        df = df[df[target_column].notna()]

    # separate features/target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    scaler = None
    # keep only numeric feature columns for modeling (drop Date/Time/object columns)
    X = X.select_dtypes(include=[np.number])

    if normalize:
        X, scaler = normalize_df(X)

    return X, y, scaler


def save_features(df: pd.DataFrame, out_path: Optional[Path] = None) -> Path:
    out_dir = DATA_DIR / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        # versioning
        existing = list(out_dir.glob("airquality_features_v*.csv"))
        max_v = 0
        for f in existing:
            name = f.stem
            m = name.rfind("_v")
            if m != -1:
                try:
                    v = int(name[m+2:])
                    if v > max_v:
                        max_v = v
                except Exception:
                    pass
        v = max_v + 1
        out_path = out_dir / f"airquality_features_v{v}.csv"
    df.to_csv(out_path, index=True)
    return out_path


if __name__ == "__main__":
    try:
        df = load_airquality_uci()
        X, y, scaler = prepare_for_model(df, target_column="C6H6(GT)")
        features = X.copy()
        features["target"] = y
        out = save_features(features)
        print(f"Saved features to: {out}")
        print("Features shape:", features.shape)
    except Exception as e:
        print("Error preparing features:", e)
