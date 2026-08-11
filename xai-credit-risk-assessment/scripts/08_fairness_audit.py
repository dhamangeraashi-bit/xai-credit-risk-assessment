"""
08_fairness_audit.py
=====================
Maps to paper section: VII-B (Fairness Across Borrower Groups), Table III,
Fig. 7.

Audits the XGBoost model's test-set predictions across two protected
attributes:
  - Sex proxy (female vs. male), from personal_status_sex
  - Age group (<25 vs. >=25)

Computes, for each attribute:
  - Demographic Parity Difference : P(approve|A) - P(approve|B)
  - Disparate Impact Ratio        : min(approval rate) / max(approval rate)
                                     ("80% rule", Feldman et al. [23])
  - Equal Opportunity Difference  : TPR gap among applicants who are truly
                                     Good credit risks (Hardt et al. [24])

"Approve" = model predicts Good (target=1). "Female" = personal_status_sex
in {A92, A95}. See src/config.py for the exact codes.

Usage:
    python scripts/02_train_models.py   # must run first
    python scripts/08_fairness_audit.py
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import AGE_THRESHOLD, DATA_PROCESSED_DIR, FEMALE_CODES, FIGURES_DIR, TABLES_DIR
from src.utils.logger import get_logger

log = get_logger("08_fairness_audit")


def group_metrics(y_true, y_pred, group_mask, group_a_name, group_b_name):
    """group_mask=True -> group A; group_mask=False -> group B."""
    approve_a = y_pred[group_mask].mean()
    approve_b = y_pred[~group_mask].mean()

    dpd = approve_a - approve_b
    dir_ = min(approve_a, approve_b) / max(approve_a, approve_b)

    # Equal opportunity: TPR among truly-Good (y_true==1) applicants, per group
    good_a = group_mask & (y_true == 1)
    good_b = (~group_mask) & (y_true == 1)
    tpr_a = y_pred[good_a].mean() if good_a.sum() > 0 else np.nan
    tpr_b = y_pred[good_b].mean() if good_b.sum() > 0 else np.nan
    eod = tpr_a - tpr_b

    return {
        f"approval_rate_{group_a_name}": round(approve_a, 3),
        f"approval_rate_{group_b_name}": round(approve_b, 3),
        "demographic_parity_diff": round(dpd, 3),
        "disparate_impact_ratio": round(dir_, 2),
        "equal_opportunity_diff": round(eod, 3),
    }


def main():
    bundle_path = os.path.join(DATA_PROCESSED_DIR, "pipeline_state.pkl")
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    xgb_model = bundle["models"]["XGBoost"]
    prepared = bundle["prepared"]
    df = bundle["df"]

    X_test = prepared.X_test_tree
    y_test = prepared.y_test.values
    y_pred = xgb_model.predict(X_test)  # 1 = approved (Good), 0 = declined (Bad)

    test_rows = df.loc[prepared.test_index]

    is_female = test_rows["personal_status_sex"].isin(FEMALE_CODES).values
    is_young = (test_rows["age"] < AGE_THRESHOLD).values

    sex_metrics = group_metrics(y_test, y_pred, is_female, "female", "male")
    age_metrics = group_metrics(y_test, y_pred, is_young, "under25", "25plus")

    table_iii = pd.DataFrame([
        {"protected_attribute": "Sex proxy (female vs. male)", **sex_metrics},
        {"protected_attribute": "Age group (<25 vs. >=25)", **age_metrics},
    ])
    print("\n" + table_iii.to_string(index=False))

    out_csv = os.path.join(TABLES_DIR, "table_III_fairness_metrics.csv")
    table_iii.to_csv(out_csv, index=False)
    log.info("Wrote %s", out_csv)

    # --- Fig. 7: approval rate by group, with 80%-rule threshold line ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    sex_rates = [sex_metrics["approval_rate_female"], sex_metrics["approval_rate_male"]]
    axes[0].bar(["Female", "Male"], sex_rates, color=["#d62728", "#1f77b4"])
    threshold_sex = max(sex_rates) * 0.8
    axes[0].axhline(threshold_sex, color="gray", linestyle="--", label="80% rule threshold")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Model approval rate")
    axes[0].set_title("By sex proxy")
    axes[0].legend()

    age_rates = [age_metrics["approval_rate_under25"], age_metrics["approval_rate_25plus"]]
    axes[1].bar(["Under 25", "25 and over"], age_rates, color=["#d62728", "#1f77b4"])
    threshold_age = max(age_rates) * 0.8
    axes[1].axhline(threshold_age, color="gray", linestyle="--", label="80% rule threshold")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("By age group")
    axes[1].legend()

    fig.suptitle("Fig. 7 - Model Approval Rate by Sex Proxy (left) and Age Group (right)")
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "fig7_fairness_approval_rates.png")
    plt.savefig(out_path, dpi=200)
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
