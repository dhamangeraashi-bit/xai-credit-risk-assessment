"""
02_train_models.py
===================
Maps to paper section: V (Methodology and Implementation), VI (Table II).

Trains Logistic Regression, Random Forest and XGBoost on the 75% training
split, evaluates all three on the 25% held-out test split, and writes
outputs/tables/table_II_model_comparison.csv.

Also persists the trained models and the prepared train/test data (as a
pickle) to data/processed/, so that downstream scripts (SHAP, LIME,
fairness, thin-file analysis) can reuse the *exact same* split and models
without retraining -- important for internal consistency of the whole
reproduction package.

Usage:
    python scripts/02_train_models.py
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_PROCESSED_DIR, TABLES_DIR
from src.data_loader import add_binary_target, load_raw_dataframe
from src.evaluate import build_metrics_table
from src.models import train_all_models
from src.preprocessing import prepare_datasets
from src.utils.logger import get_logger

log = get_logger("02_train_models")


def main():
    log.info("Loading dataset...")
    df = add_binary_target(load_raw_dataframe())

    log.info("Preprocessing (label encoding, 75/25 stratified split, scaling)...")
    prepared = prepare_datasets(df)

    log.info("Training Logistic Regression, Random Forest, XGBoost...")
    models = train_all_models(prepared.X_train_tree, prepared.X_train_linear, prepared.y_train)

    X_test_map = {
        "Logistic Regression": prepared.X_test_linear,
        "Random Forest": prepared.X_test_tree,
        "XGBoost": prepared.X_test_tree,
    }

    log.info("Evaluating on held-out test set (Table II)...")
    table_ii = build_metrics_table(models, X_test_map, prepared.y_test)
    print("\n" + table_ii.to_string())

    out_path = os.path.join(TABLES_DIR, "table_II_model_comparison.csv")
    table_ii.to_csv(out_path)
    log.info("Wrote %s", out_path)

    # Persist everything downstream scripts need.
    bundle_path = os.path.join(DATA_PROCESSED_DIR, "pipeline_state.pkl")
    with open(bundle_path, "wb") as f:
        pickle.dump({"models": models, "prepared": prepared, "df": df}, f)
    log.info("Persisted trained models + split to %s", bundle_path)


if __name__ == "__main__":
    main()
