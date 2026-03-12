from __future__ import annotations

from typing import Tuple

import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Dataset is empty.")
    return df


def split_features_target(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")

    feature_df = df.drop(columns=[target])
    feature_df = feature_df.select_dtypes(include=["number"])
    if feature_df.shape[1] == 0:
        raise ValueError("No numeric feature columns found after filtering.")

    y = df[target]
    return feature_df, y
