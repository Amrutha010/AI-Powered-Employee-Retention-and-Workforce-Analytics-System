from __future__ import annotations

from typing import Tuple

import pandas as pd

from .config import DROP_COLUMNS, TARGET_COLUMN


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in dataset.")
    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    cleaned = df.drop(columns=[col for col in DROP_COLUMNS if col in df.columns], errors="ignore")
    y = cleaned[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    if y.isna().any():
        raise ValueError("Target column contains values other than 'Yes'/'No'.")
    X = cleaned.drop(columns=[TARGET_COLUMN])
    return X, y.astype(int)
