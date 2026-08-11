"""
07_shap_vs_lime.py
===================
Maps to paper section: VII-A (SHAP vs. LIME: Explanation Quality
Comparison), Fig. 6.

For 40 held-out test applicants (matching the paper's stated sample size),
computes:
  - SHAP feature attributions (TreeExplainer, already fast/exact)
  - LIME feature attributions (perturbation-based local surrogate)
per applicant, then:
  - Spearman rank correlation between |SHAP| and |LIME| feature importances
  - Top-3 feature overlap
  - Per-instance compute time for each method

Usage:
    python scripts/02_train_models.py   # must run first
    python scripts/07_shap_vs_lime.py
"""

import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_PROCESSED_DIR, FIGURES_DIR, RANDOM_SEED, TABLES_DIR
from src.utils.logger import get_logger

log = get_logger("07_shap_vs_lime")

N_APPLICANTS = 40  # paper: "held-out test set of 40 credit applicants" (Sec. VII-A)


def main():
    bundle_path = os.path.join(DATA_PROCESSED_DIR, "pipeline_state.pkl")
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    xgb_model = bundle["models"]["XGBoost"]
    prepared = bundle["prepared"]
    X_train, X_test = prepared.X_train_tree, prepared.X_test_tree
    feature_names = prepared.feature_names

    rng = np.random.RandomState(RANDOM_SEED)
    sample_idx = rng.choice(len(X_test), size=N_APPLICANTS, replace=False)
    X_sample = X_test.iloc[sample_idx]

    # --- SHAP ---
    log.info("Computing SHAP explanations for %d applicants...", N_APPLICANTS)
    explainer_shap = shap.TreeExplainer(xgb_model)
    t0 = time.perf_counter()
    shap_vals = explainer_shap.shap_values(X_sample)
    shap_total_time = time.perf_counter() - t0
    shap_per_instance_ms = (shap_total_time / N_APPLICANTS) * 1000

    # --- LIME ---
    log.info("Computing LIME explanations for %d applicants (this is slower by design)...", N_APPLICANTS)
    explainer_lime = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=["Bad", "Good"],
        mode="classification",
        random_state=RANDOM_SEED,
        discretize_continuous=True,
    )

    def predict_fn(x):
        return xgb_model.predict_proba(pd.DataFrame(x, columns=feature_names))

    lime_importances = []
    lime_times = []
    for i in range(N_APPLICANTS):
        row = X_sample.iloc[i].values
        t0 = time.perf_counter()
        exp = explainer_lime.explain_instance(row, predict_fn, num_features=len(feature_names))
        lime_times.append(time.perf_counter() - t0)
        # Map LIME's (condition-string -> weight) back onto feature order
        weights = dict.fromkeys(feature_names, 0.0)
        for condition, weight in exp.as_list():
            for feat in feature_names:
                if feat in condition:
                    weights[feat] = weight
                    break
        lime_importances.append([weights[f] for f in feature_names])

    lime_per_instance_ms = (sum(lime_times) / N_APPLICANTS) * 1000
    lime_importances = np.array(lime_importances)

    # --- Agreement metrics ---
    correlations, overlaps = [], []
    for i in range(N_APPLICANTS):
        shap_abs = np.abs(shap_vals[i])
        lime_abs = np.abs(lime_importances[i])
        rho, _ = spearmanr(shap_abs, lime_abs)
        correlations.append(rho if not np.isnan(rho) else 0.0)

        top3_shap = set(np.argsort(-shap_abs)[:3])
        top3_lime = set(np.argsort(-lime_abs)[:3])
        overlaps.append(len(top3_shap & top3_lime) / 3.0)

    mean_rho, sd_rho = float(np.mean(correlations)), float(np.std(correlations))
    mean_overlap = float(np.mean(overlaps))
    speedup = lime_per_instance_ms / shap_per_instance_ms

    summary = pd.DataFrame([{
        "n_applicants": N_APPLICANTS,
        "mean_spearman_rho": round(mean_rho, 3),
        "sd_spearman_rho": round(sd_rho, 3),
        "mean_top3_overlap": round(mean_overlap, 3),
        "shap_ms_per_instance": round(shap_per_instance_ms, 4),
        "lime_ms_per_instance": round(lime_per_instance_ms, 2),
        "speedup_shap_over_lime": round(speedup, 1),
    }])
    print("\n" + summary.to_string(index=False))

    out_csv = os.path.join(TABLES_DIR, "shap_vs_lime_summary.csv")
    summary.to_csv(out_csv, index=False)
    log.info("Wrote %s", out_csv)

    # --- Fig. 6: two-panel figure ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))

    axes[0].hist(correlations, bins=12, color="#1f77b4", edgecolor="white")
    axes[0].axvline(mean_rho, color="red", linestyle="--", label=f"mean = {mean_rho:.2f}")
    axes[0].set_title("Per-applicant SHAP-LIME\nSpearman agreement")
    axes[0].set_xlabel("Spearman ρ")
    axes[0].set_ylabel("Number of applicants")
    axes[0].legend()

    axes[1].bar(["SHAP", "LIME"], [shap_per_instance_ms, lime_per_instance_ms],
                color=["#1f77b4", "#d62728"])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Mean explanation time per instance (ms, log scale)")
    axes[1].set_title("Explanation compute time")
    for i, v in enumerate([shap_per_instance_ms, lime_per_instance_ms]):
        axes[1].text(i, v, f"{v:.2f} ms", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Fig. 6 - SHAP vs. LIME: Agreement (left) and Speed (right)")
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "fig6_shap_vs_lime.png")
    plt.savefig(out_path, dpi=200)
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
