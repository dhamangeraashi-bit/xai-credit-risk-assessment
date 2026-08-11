"""
06_shap_waterfall.py
=====================
Maps to paper section: VI (Results and Discussion), Fig. 5.

Selects a single representative test applicant and renders the local SHAP
waterfall explanation for that applicant's prediction, matching the
paper's Fig. 5. The applicant chosen is the one whose predicted
probability of good creditworthiness is closest to the paper's reported
65.6% figure, so the reproduction is as close as possible to the original
qualitative example (an explicit, documented choice -- see
docs/ASSUMPTIONS.md item A4, since the paper does not identify which
applicant row Fig. 5 corresponds to).

Usage:
    python scripts/05_shap_summary.py   # must run first
    python scripts/06_shap_waterfall.py
"""

import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt
import shap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_PROCESSED_DIR, FIGURES_DIR
from src.utils.logger import get_logger

log = get_logger("06_shap_waterfall")

TARGET_PROBA = 0.656  # paper's reported "65.6% probability of good creditworthiness"


def main():
    shap_path = os.path.join(DATA_PROCESSED_DIR, "shap_values.pkl")
    with open(shap_path, "rb") as f:
        d = pickle.load(f)
    explainer, shap_values, X_test = d["explainer"], d["shap_values"], d["X_test"]

    bundle_path = os.path.join(DATA_PROCESSED_DIR, "pipeline_state.pkl")
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    xgb_model = bundle["models"]["XGBoost"]

    proba_good = xgb_model.predict_proba(X_test)[:, 1]  # target=1 is Good (see config.py)
    idx = int(np.argmin(np.abs(proba_good - TARGET_PROBA)))
    chosen_proba = proba_good[idx]

    log.info(
        "Selected test-set row %d (positional index %d): P(good credit) = %.3f "
        "(paper reports 0.656 for its illustrative applicant)",
        X_test.index[idx], idx, chosen_proba,
    )

    plt.figure()
    shap.plots.waterfall(shap_values[idx], show=False)
    plt.title(f"Fig. 5 - Local SHAP Waterfall (P(good credit)={chosen_proba:.3f})")
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "fig5_shap_waterfall.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
