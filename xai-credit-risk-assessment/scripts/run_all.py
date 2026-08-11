"""
run_all.py
==========
Runs the entire reproduction pipeline end to end, in the correct order,
regenerating every table and every figure in the paper from scratch.

Usage:
    python scripts/run_all.py

Equivalent to running scripts 02 through 10 in numeric order (01 is
optional / only needed to refresh the bundled dataset).
"""

import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PIPELINE = [
    "02_train_models.py",
    "03_generate_roc_curves.py",
    "04_generate_metrics_comparison.py",
    "05_shap_summary.py",
    "06_shap_waterfall.py",
    "07_shap_vs_lime.py",
    "08_fairness_audit.py",
    "09_thin_file_alt_data.py",
    "10_system_architecture_diagram.py",
]


def main():
    for script in PIPELINE:
        path = os.path.join(SCRIPT_DIR, script)
        print(f"\n{'=' * 70}\nRUNNING: {script}\n{'=' * 70}")
        result = subprocess.run([sys.executable, path])
        if result.returncode != 0:
            print(f"\nFAILED at {script} (exit code {result.returncode}). Stopping.")
            sys.exit(result.returncode)
    print("\nAll scripts completed successfully.")
    print("Tables  -> outputs/tables/")
    print("Figures -> outputs/figures/")
    print("Logs    -> outputs/logs/")
    print("Dashboard (Fig. 9, open in any browser) -> dashboard/dashboard.html")


if __name__ == "__main__":
    main()
