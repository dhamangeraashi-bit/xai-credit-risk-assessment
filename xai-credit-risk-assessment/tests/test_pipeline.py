"""
test_pipeline.py
=================
Lightweight smoke tests -- not a full test suite, but enough to catch
"the pipeline is broken" before a reviewer does.

Usage:
    python -m pytest tests/ -v
    (or simply: python tests/test_pipeline.py)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import add_binary_target, load_raw_dataframe
from src.preprocessing import prepare_datasets
from src.models import build_logistic_regression, build_random_forest, build_xgboost


def test_dataset_shape():
    df = load_raw_dataframe()
    assert df.shape == (1000, 21), f"Expected (1000, 21), got {df.shape}"


def test_class_balance():
    df = add_binary_target(load_raw_dataframe())
    counts = df["target"].value_counts().to_dict()
    assert counts[1] == 700, f"Expected 700 Good (target=1), got {counts.get(1)}"
    assert counts[0] == 300, f"Expected 300 Bad (target=0), got {counts.get(0)}"


def test_preprocessing_split_sizes():
    df = add_binary_target(load_raw_dataframe())
    prepared = prepare_datasets(df)
    assert len(prepared.X_train_tree) == 750
    assert len(prepared.X_test_tree) == 250
    assert len(prepared.X_train_tree) == len(prepared.X_train_linear)
    assert len(prepared.X_test_tree) == len(prepared.X_test_linear)


def test_models_fit_and_predict():
    df = add_binary_target(load_raw_dataframe())
    prepared = prepare_datasets(df)

    lr = build_logistic_regression().fit(prepared.X_train_linear, prepared.y_train)
    rf = build_random_forest().fit(prepared.X_train_tree, prepared.y_train)
    xgb = build_xgboost().fit(prepared.X_train_tree, prepared.y_train)

    for name, model, X_test in [
        ("LogReg", lr, prepared.X_test_linear),
        ("RandomForest", rf, prepared.X_test_tree),
        ("XGBoost", xgb, prepared.X_test_tree),
    ]:
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)
        assert len(preds) == 250, f"{name}: expected 250 predictions"
        assert proba.shape == (250, 2), f"{name}: expected (250, 2) proba shape"
        assert set(preds.tolist()).issubset({0, 1}), f"{name}: predictions must be binary"


if __name__ == "__main__":
    tests = [test_dataset_shape, test_class_balance, test_preprocessing_split_sizes, test_models_fit_and_predict]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll smoke tests passed.")
