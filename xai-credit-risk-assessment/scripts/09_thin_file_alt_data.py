"""
09_thin_file_alt_data.py
=========================
Maps to paper section: VII-C (Thin-File Borrowers: Does Alternative Data
Help Where It Is Needed Most?), Table IV, Fig. 8.

IMPORTANT -- SYNTHETIC DATA DISCLOSURE (see docs/ASSUMPTIONS.md item A5 and
the paper's own Section VIII "Limitations"):
The German Credit dataset contains NO real mobile-recharge, utility-payment,
or digital-transaction-consistency data. The paper explicitly states these
alternative-data features are "engineered" / "simulated" to a realistic but
synthetic signal strength (~0.70 standalone AUC). This script reproduces
that same explicit design choice -- it does NOT claim these are real
alternative-data signals. The three features are generated as noisy
functions of the true label (with the label itself withheld from any
model), calibrated so that a simple bureau-free classifier trained on them
alone reaches approximately 0.70 standalone ROC-AUC, matching the paper's
stated calibration target. This is a documented simulation, not invented
results: the downstream comparison (bureau-only vs bureau+alt-data AUC,
split by thin-file/banked) is computed honestly from this simulated input.

Usage:
    python scripts/02_train_models.py   # must run first
    python scripts/09_thin_file_alt_data.py
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_PROCESSED_DIR, FIGURES_DIR, RANDOM_SEED, TABLES_DIR, THIN_FILE_CODE
from src.models import build_xgboost
from src.utils.logger import get_logger

log = get_logger("09_thin_file_alt_data")

NOISE_SIGMA = 4.3  # calibrated empirically so standalone alt-data AUC ~= 0.70 (see calibration check below)


def engineer_alt_data_features(y: np.ndarray, rng: np.random.RandomState) -> pd.DataFrame:
    """
    Generate 3 synthetic alternative-data features correlated with the true
    label y (1=Good, 0=Bad), calibrated to a moderate standalone signal.
    """
    n = len(y)
    latent = np.where(y == 1, 1.0, -1.0)  # +1 for Good, -1 for Bad

    mobile_recharge_regularity = latent + rng.normal(0, NOISE_SIGMA, n)
    utility_payment_punctuality = latent + rng.normal(0, NOISE_SIGMA, n)
    digital_txn_consistency = latent + rng.normal(0, NOISE_SIGMA, n)

    def to_unit_scale(x):
        x = (x - x.min()) / (x.max() - x.min())
        return x

    return pd.DataFrame({
        "mobile_recharge_regularity": to_unit_scale(mobile_recharge_regularity),
        "utility_payment_punctuality": to_unit_scale(utility_payment_punctuality),
        "digital_txn_consistency": to_unit_scale(digital_txn_consistency),
    })


def main():
    bundle_path = os.path.join(DATA_PROCESSED_DIR, "pipeline_state.pkl")
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    prepared = bundle["prepared"]
    df = bundle["df"]

    rng = np.random.RandomState(RANDOM_SEED)
    y_full = df["target"].values
    alt_features_full = engineer_alt_data_features(y_full, rng)
    alt_features_full.index = df.index

    # Calibration check: standalone AUC of the 3 alt-data features alone (whole dataset)
    standalone_auc = roc_auc_score(
        y_full,
        LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        .fit(alt_features_full, y_full)
        .predict_proba(alt_features_full)[:, 1],
    )
    log.info("Calibration check: standalone alt-data AUC = %.3f (paper target: ~0.70)", standalone_auc)

    # Build bureau-only and bureau+alt-data train/test feature sets
    X_train_bureau = prepared.X_train_tree
    X_test_bureau = prepared.X_test_tree
    X_train_alt = alt_features_full.loc[prepared.train_index]
    X_test_alt = alt_features_full.loc[prepared.test_index]

    X_train_combined = pd.concat([X_train_bureau, X_train_alt], axis=1)
    X_test_combined = pd.concat([X_test_bureau, X_test_alt], axis=1)

    y_train, y_test = prepared.y_train, prepared.y_test

    log.info("Training bureau-only XGBoost model...")
    model_bureau = build_xgboost().fit(X_train_bureau, y_train)

    log.info("Training bureau+alt-data XGBoost model...")
    model_combined = build_xgboost().fit(X_train_combined, y_train)

    # Identify thin-file (no checking account, A14) vs banked applicants in test set
    test_rows = df.loc[prepared.test_index]
    is_thin_file = (test_rows["checking_account_status"] == THIN_FILE_CODE).values

    results = []
    for label, mask in [("Thin-file (no checking a/c)", is_thin_file), ("Banked (has checking a/c)", ~is_thin_file)]:
        n_sub = mask.sum()
        y_sub = y_test.values[mask]

        proba_bureau = model_bureau.predict_proba(X_test_bureau[mask])[:, 1]
        proba_combined = model_combined.predict_proba(X_test_combined[mask])[:, 1]

        auc_bureau = roc_auc_score(y_sub, proba_bureau)
        auc_combined = roc_auc_score(y_sub, proba_combined)

        results.append({
            "subgroup": f"{label}, n={n_sub}",
            "bureau_only_auc": round(auc_bureau, 3),
            "bureau_plus_altdata_auc": round(auc_combined, 3),
            "auc_gain": round(auc_combined - auc_bureau, 3),
        })

    table_iv = pd.DataFrame(results)
    print("\n" + table_iv.to_string(index=False))

    out_csv = os.path.join(TABLES_DIR, "table_IV_thin_file_alt_data.csv")
    table_iv.to_csv(out_csv, index=False)
    log.info("Wrote %s", out_csv)

    # --- Fig. 8 ---
    fig, ax = plt.subplots(figsize=(7, 5.5))
    x = np.arange(len(table_iv))
    width = 0.35
    ax.bar(x - width / 2, table_iv["bureau_only_auc"], width, label="Bureau-only", color="#7f7f7f")
    ax.bar(x + width / 2, table_iv["bureau_plus_altdata_auc"], width, label="Bureau + Alt-data", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(table_iv["subgroup"], rotation=10)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0, 1)
    ax.set_title("Fig. 8 - ROC-AUC With/Without Alternative-Data Features,\nby Thin-File vs. Banked Subgroup")
    ax.legend()
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "fig8_thin_file_altdata.png")
    plt.savefig(out_path, dpi=200)
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
