"""Preprocessing utilities: missing values, normalization, train/test split helpers."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def handle_missing_values(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    df = df.copy()
    # numeric columns: fill with mean/median/zero
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if strategy == "mean":
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    elif strategy == "median":
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    else:
        df[num_cols] = df[num_cols].fillna(0)

    # non-numeric: fill with mode if available, else forward-fill then empty string
    obj_cols = [c for c in df.columns if c not in num_cols]
    for c in obj_cols:
        if df[c].isnull().any():
            try:
                mode = df[c].mode(dropna=True)
                if len(mode) > 0:
                    df[c] = df[c].fillna(mode[0])
                else:
                    df[c] = df[c].fillna("")
            except Exception:
                df[c] = df[c].fillna("")

    return df


def normalize_df(df: pd.DataFrame, columns=None):
    scaler = StandardScaler()
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    df[cols] = scaler.fit_transform(df[cols])
    return df, scaler


def split_features_target(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
