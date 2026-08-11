"""
04_generate_metrics_comparison.py
==================================
Maps to paper section: VI (Results and Discussion), Fig. 3.

Renders Table II (all 5 metrics x 3 models) as a grouped bar chart.

Usage:
    python scripts/04_generate_metrics_comparison.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import FIGURES_DIR, TABLES_DIR
from src.utils.logger import get_logger

log = get_logger("04_generate_metrics_comparison")


def main():
    table_path = os.path.join(TABLES_DIR, "table_II_model_comparison.csv")
    df = pd.read_csv(table_path, index_col="Model")

    metrics = df.columns.tolist()
    models = df.index.tolist()
    x = np.arange(len(metrics))
    width = 0.25

    plt.figure(figsize=(8, 5))
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    for i, model in enumerate(models):
        plt.bar(x + i * width, df.loc[model].values, width, label=model, color=colors[i])

    plt.xticks(x + width, metrics)
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title("Fig. 3 - Evaluation Metrics Across the Three Models")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "fig3_metrics_comparison.png")
    plt.savefig(out_path, dpi=200)
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
