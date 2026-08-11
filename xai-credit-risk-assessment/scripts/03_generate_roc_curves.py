"""
03_generate_roc_curves.py
==========================
Maps to paper section: VI (Results and Discussion), Fig. 2.

Loads the trained models from data/processed/pipeline_state.pkl (produced
by scripts/02_train_models.py) and plots ROC curves for all three
classifiers on the held-out test set.

Usage:
    python scripts/02_train_models.py   # must run first
    python scripts/03_generate_roc_curves.py
"""

import os
import pickle
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_PROCESSED_DIR, FIGURES_DIR
from src.evaluate import get_roc_points
from src.utils.logger import get_logger

log = get_logger("03_generate_roc_curves")


def main():
    bundle_path = os.path.join(DATA_PROCESSED_DIR, "pipeline_state.pkl")
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    models, prepared = bundle["models"], bundle["prepared"]

    X_test_map = {
        "Logistic Regression": prepared.X_test_linear,
        "Random Forest": prepared.X_test_tree,
        "XGBoost": prepared.X_test_tree,
    }
    points = get_roc_points(models, X_test_map, prepared.y_test)

    plt.figure(figsize=(6.2, 5.6))
    colors = {"Logistic Regression": "#1f77b4", "Random Forest": "#2ca02c", "XGBoost": "#d62728"}
    for name, (fpr, tpr, auc) in points.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=colors[name], linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Fig. 2 - ROC Curves:\nLogistic Regression vs Random Forest vs XGBoost", fontsize=11)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "fig2_roc_curves.png")
    plt.savefig(out_path, dpi=200)
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
