# Figure / Table -> Paper Section Mapping

| Paper item | Paper section | Script(s) | Output file(s) |
|---|---|---|---|
| Fig. 1 (pipeline diagram) | III-B System Architecture | `scripts/10_system_architecture_diagram.py` | `outputs/figures/fig1_system_architecture.png` |
| Table I (related work) | IV Related Work | *(narrative table, not computational — see note below)* | — |
| Table II (model comparison) | VI Results and Discussion | `scripts/02_train_models.py` | `outputs/tables/table_II_model_comparison.csv` |
| Fig. 2 (ROC curves) | VI Results and Discussion | `scripts/03_generate_roc_curves.py` | `outputs/figures/fig2_roc_curves.png` |
| Fig. 3 (metrics bar chart) | VI Results and Discussion | `scripts/04_generate_metrics_comparison.py` | `outputs/figures/fig3_metrics_comparison.png` |
| Fig. 4 (global SHAP summary) | VI Results and Discussion | `scripts/05_shap_summary.py` | `outputs/figures/fig4_shap_summary.png` |
| Fig. 5 (local SHAP waterfall) | VI Results and Discussion | `scripts/06_shap_waterfall.py` | `outputs/figures/fig5_shap_waterfall.png` |
| Fig. 6 (SHAP vs LIME) | VII-A Explanation Quality | `scripts/07_shap_vs_lime.py` | `outputs/figures/fig6_shap_vs_lime.png`, `outputs/tables/shap_vs_lime_summary.csv` |
| Table III (fairness metrics) | VII-B Fairness Across Groups | `scripts/08_fairness_audit.py` | `outputs/tables/table_III_fairness_metrics.csv` |
| Fig. 7 (approval-rate by group) | VII-B Fairness Across Groups | `scripts/08_fairness_audit.py` | `outputs/figures/fig7_fairness_approval_rates.png` |
| Table IV (thin-file AUC gain) | VII-C Thin-File Borrowers | `scripts/09_thin_file_alt_data.py` | `outputs/tables/table_IV_thin_file_alt_data.csv` |
| Fig. 8 (thin-file vs banked AUC) | VII-C Thin-File Borrowers | `scripts/09_thin_file_alt_data.py` | `outputs/figures/fig8_thin_file_altdata.png` |
| Fig. 9 (loan-officer dashboard) | VII-D Dashboard | `dashboard/dashboard.html` (static prototype, no generation script needed) | `outputs/figures/fig9_dashboard_screenshot.png` (reference screenshot) |

**Note on Table I:** Table I is a narrative literature-comparison table
(author, focus area, limitation, contribution) with no computational
content — it summarizes related work, not experimental results, so there is
no script to "reproduce" it. It is reproduced verbatim in
`docs/paper_reference/table_I.md` for completeness only.

## Recommended run order

`scripts/run_all.py` runs everything in the correct dependency order
automatically. If running scripts individually, this order matters:

1. `01_download_data.py` (optional — data is already bundled)
2. `02_train_models.py` — **must run first**; persists trained models + split to `data/processed/pipeline_state.pkl`, which every later script loads
3. `03_generate_roc_curves.py`
4. `04_generate_metrics_comparison.py`
5. `05_shap_summary.py` — persists SHAP values to `data/processed/shap_values.pkl`, used by script 06
6. `06_shap_waterfall.py`
7. `07_shap_vs_lime.py`
8. `08_fairness_audit.py`
9. `09_thin_file_alt_data.py`
10. `10_system_architecture_diagram.py`
