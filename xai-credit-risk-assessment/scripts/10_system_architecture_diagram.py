"""
10_system_architecture_diagram.py
==================================
Maps to paper section: III-B (System Architecture), Fig. 1.

Renders the four-layer pipeline diagram (Data -> Model -> Explainability ->
Decision) described in Section III-B, purely programmatically (matplotlib
shapes), so it is regenerated from code rather than hand-drawn.

Usage:
    python scripts/10_system_architecture_diagram.py
"""

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrow

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import FIGURES_DIR
from src.utils.logger import get_logger

log = get_logger("10_system_architecture_diagram")

LAYERS = [
    ("Data Layer", "Ingest applicant attributes:\nincome, age, employment,\nloan amount, credit history,\nexisting debts, dependents", "#dbe9f7"),
    ("Model Layer", "Train & compare:\nLogistic Regression\nRandom Forest\nXGBoost", "#d9f0da"),
    ("Explainability Layer", "Apply SHAP (TreeExplainer)\nto the selected model", "#fdecd2"),
    ("Decision Layer", "Combine prediction + SHAP\nattributions into\nApprove/Reject + reason codes", "#f6d6d6"),
]


def main():
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    n = len(LAYERS)
    box_w, box_h = 1.9, 2.6
    gap = 0.7
    total_w = n * box_w + (n - 1) * gap
    start_x = -total_w / 2

    for i, (title, desc, color) in enumerate(LAYERS):
        x = start_x + i * (box_w + gap)
        box = FancyBboxPatch(
            (x, -box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.05,rounding_size=0.12",
            linewidth=1.5, edgecolor="#333333", facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x + box_w / 2, box_h / 2 - 0.35, title, ha="center", va="top",
                fontsize=10.5, fontweight="bold", wrap=True)
        ax.text(x + box_w / 2, -0.15, desc, ha="center", va="center", fontsize=8.2)

        if i < n - 1:
            arrow_x0 = x + box_w + 0.05
            ax.annotate("", xy=(arrow_x0 + gap - 0.1, 0), xytext=(arrow_x0, 0),
                        arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.8))

    ax.set_xlim(start_x - 0.5, -start_x + 0.5)
    ax.set_ylim(-box_h / 2 - 0.6, box_h / 2 + 0.6)
    ax.axis("off")
    ax.set_title("Fig. 1 - Four-Layer Explainable Credit-Risk Assessment Pipeline", fontsize=12, pad=15)
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "fig1_system_architecture.png")
    plt.savefig(out_path, dpi=200)
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
