"""
preprocessing.py
================
Maps to paper section: V-C (Experimental Setup) and III-C (Key Modules -
Preprocessing module).

- Categorical attributes -> label-encoded integers ("labels in coding").
- Numerical attributes   -> kept raw for tree models, standardised (z-score)
  for Logistic Regression, matching the paper's stated design.
- Split                  -> 75/25 stratified train/test split.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS, RANDOM_SEED, TEST_SIZE


@dataclass
class PreparedData:
    X_train_tree: pd.DataFrame     # for RF / XGBoost (raw numerics, label-encoded categoricals)
    X_test_tree: pd.DataFrame
    X_train_linear: pd.DataFrame   # for Logistic Regression (numerics standardised)
    X_test_linear: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_index: pd.Index
    test_index: pd.Index
    feature_names: list


def label_encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all categorical columns in place, return a new frame."""
    df = df.copy()
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def prepare_datasets(df: pd.DataFrame) -> PreparedData:
    """
    df must already contain the binary `target` column (see data_loader.add_binary_target).
    Returns train/test splits, both in "tree-ready" and "linear-ready" form,
    sharing the same row split and the same label-encoded categorical columns.
    """
    feature_cols = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS
    encoded = label_encode_categoricals(df)

    X = encoded[feature_cols]
    y = encoded["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )

    # Tree-ready views: exactly X_train / X_test (raw numerics, encoded categoricals)
    X_train_tree = X_train.copy()
    X_test_tree = X_test.copy()

    # Linear-ready views: standardise numeric columns only, fit on train only
    scaler = StandardScaler()
    X_train_linear = X_train.copy()
    X_test_linear = X_test.copy()
    X_train_linear[NUMERICAL_COLUMNS] = scaler.fit_transform(X_train[NUMERICAL_COLUMNS])
    X_test_linear[NUMERICAL_COLUMNS] = scaler.transform(X_test[NUMERICAL_COLUMNS])

    return PreparedData(
        X_train_tree=X_train_tree,
        X_test_tree=X_test_tree,
        X_train_linear=X_train_linear,
        X_test_linear=X_test_linear,
        y_train=y_train,
        y_test=y_test,
        train_index=X_train.index,
        test_index=X_test.index,
        feature_names=feature_cols,
    )
