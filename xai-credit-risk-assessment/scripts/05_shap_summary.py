"""
05_shap_summary.py
===================
Maps to paper section: VI (Results and Discussion), Fig. 4.

Applies SHAP's TreeExplainer to the selected best-performing model
(XGBoost, per Table II / Section VI) and renders the global SHAP summary
(beeswarm) plot across the full held-out test set.

Usage:
    python scripts/05_shap_summary.py
"""

import os
import pickle
import sys

import matplotlib.pyplot as plt
import shap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_PROCESSED_DIR, FIGURES_DIR
from src.utils.logger import get_logger

log = get_logger("05_shap_summary")


def main():
    bundle_path = os.path.join(DATA_PROCESSED_DIR, "pipeline_state.pkl")
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    xgb_model = bundle["models"]["XGBoost"]
    X_test = bundle["prepared"].X_test_tree

    log.info("Building SHAP TreeExplainer for XGBoost...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_test)

    # Persist raw SHAP values for reuse by script 06 (waterfall) and 07 (vs LIME)
    shap_path = os.path.join(DATA_PROCESSED_DIR, "shap_values.pkl")
    with open(shap_path, "wb") as f:
        pickle.dump({"explainer": explainer, "shap_values": shap_values, "X_test": X_test}, f)
    log.info("Persisted SHAP values to %s", shap_path)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("Fig. 4 - Global SHAP Summary: Feature Impact on Model Output")
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "fig4_shap_summary.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
