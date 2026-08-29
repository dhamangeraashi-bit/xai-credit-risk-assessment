

## 1. Multi-seed stability analysis
Re-run `scripts/run_all.py` across 10-20 random seeds and report mean ±
standard deviation for every cell of Table II, Table III, and Table IV.
This would directly address the fairness-direction discrepancy noted in
`docs/ASSUMPTIONS.md` (A7) by showing whether the paper's reported
equal-opportunity gap direction is stable or itself seed-sensitive.

## 2. Hyperparameter search
The paper (and this reproduction) fixes model hyperparameters by
reasonable default rather than tuning. A grid or Bayesian search
(`GridSearchCV`/`Optuna`) over XGBoost's `max_depth`, `learning_rate`,
`n_estimators` could report a best-achievable Table II, with the current
numbers kept as the "default configuration" baseline.

## 3. k-fold cross-validation
Replace the single 75/25 stratified split with 5- or 10-fold stratified
cross-validation, reporting mean ± std for every metric. This would give a
confidence interval around each cell of Table II rather than a single
point estimate.

## 4. Larger SHAP-vs-LIME comparison sample
Section VII-A uses 40 applicants "for tractability given LIME's per-instance
cost" (paper's own words). Re-running `scripts/07_shap_vs_lime.py` with
`N_APPLICANTS` increased to the full ~250-row test set (LIME will take
several minutes longer) would tighten the confidence interval on the
reported Spearman correlation and top-3 overlap.

## 5. Intersectional fairness auditing
Section VIII explicitly flags this as future work: audit intersectional
subgroups (e.g., young female thin-file applicants) rather than only the
two single-attribute splits in Table III. `scripts/08_fairness_audit.py`'s
`group_metrics()` function is written generically enough to accept any
boolean mask, so this is a matter of constructing the intersectional mask
and calling it again.

## 6. Real alternative-data validation
The single biggest limitation flagged by the paper itself (Section VIII):
validate Section VII-C's findings against real Indian telecom/UPI/utility
data under a data-sharing agreement with a fintech NBFC, replacing the
synthetic features in `scripts/09_thin_file_alt_data.py` with real ones of
the same schema.

## 7. Calibration analysis
Add reliability diagrams / Brier scores for all three models — useful for a
credit-risk context where predicted probabilities (not just classifications)
often feed directly into pricing decisions, but not currently reported by
the paper.

## 8. SHAP interaction values
`shap.TreeExplainer` supports pairwise SHAP interaction values
(`explainer.shap_interaction_values(X)`), which could surface second-order
effects (e.g., does credit history matter more when checking-account status
is also poor?) beyond the additive attributions in Fig. 4.

## 9. Human user study
Section VIII notes that "a human user study with actual loan officers would
strengthen the comparison" between SHAP and LIME explanation quality — this
is a wet-lab/survey study outside the scope of a code reproduction, but is
flagged here as the paper's own stated highest-value follow-up.
