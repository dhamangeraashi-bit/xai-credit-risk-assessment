"""
evaluate.py
===========
Maps to paper section: VI (Results and Discussion), Table II, Figs. 2-3.

Computes Accuracy, Precision, Recall, F1-score and ROC-AUC for each model
on the held-out test set, with the positive class = "Bad / at-risk"
(target=1), consistent with src/config.py's target convention.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 3),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 3),
        "F1-score": round(f1_score(y_test, y_pred, zero_division=0), 3),
        "ROC-AUC": round(roc_auc_score(y_test, y_proba), 3),
    }


def build_metrics_table(models: dict, X_test_map: dict, y_test) -> pd.DataFrame:
    """
    models: {"Logistic Regression": fitted_model, ...}
    X_test_map: {"Logistic Regression": X_test_linear, "Random Forest": X_test_tree, ...}
    """
    rows = [evaluate_model(model, X_test_map[name], y_test, name) for name, model in models.items()]
    return pd.DataFrame(rows).set_index("Model")


def get_roc_points(models: dict, X_test_map: dict, y_test) -> dict:
    """Return {"model_name": (fpr, tpr, auc)} for ROC-curve plotting (Fig. 2)."""
    points = {}
    for name, model in models.items():
        proba = model.predict_proba(X_test_map[name])[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        points[name] = (fpr, tpr, auc)
    return points
