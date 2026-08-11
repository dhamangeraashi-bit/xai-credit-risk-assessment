"""
models.py
=========
Maps to paper section: V-C (Experimental Setup), III-C (Model layer).

Trains the three candidate classifiers with the imbalance-handling
strategy stated explicitly in the paper:
  - Logistic Regression, Random Forest: class_weight="balanced"
  - XGBoost: scale_pos_weight=3

ASSUMPTION (docs/ASSUMPTIONS.md, item A2): the paper does not publish
hyperparameters beyond the imbalance-handling settings above (e.g. number
of trees, tree depth, learning rate). Reasonable, commonly used defaults
are fixed below and documented so the run is reproducible; a grid search
is offered as an optional extension (see docs/OPTIONAL_EXTENSIONS.md).
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import RANDOM_SEED


def build_logistic_regression() -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_SEED,
    )


def build_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,        # ASSUMPTION: not specified in paper
        max_depth=None,          # ASSUMPTION: not specified in paper
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def build_xgboost() -> XGBClassifier:
    # NOTE on scale_pos_weight=3: this value is stated explicitly in the
    # paper (Sec. V-C) and is applied literally here to class 1 of our
    # target (Good), per the convention documented in src/config.py
    # (ASSUMPTION A3). We keep the paper's literal parameter value rather
    # than "correcting" its direction, since our goal is to reproduce the
    # paper's stated methodology, not to redesign it.
    return XGBClassifier(
        n_estimators=300,        # ASSUMPTION: not specified in paper
        max_depth=4,             # ASSUMPTION: not specified in paper
        learning_rate=0.05,      # ASSUMPTION: not specified in paper
        scale_pos_weight=3,      # stated explicitly in the paper (Sec. V-C)
        random_state=RANDOM_SEED,
        eval_metric="logloss",
        n_jobs=-1,
    )


def train_all_models(X_train_tree, X_train_linear, y_train):
    """Fit LR on the standardised view, RF/XGBoost on the raw tree-ready view."""
    lr = build_logistic_regression().fit(X_train_linear, y_train)
    rf = build_random_forest().fit(X_train_tree, y_train)
    xgb = build_xgboost().fit(X_train_tree, y_train)
    return {"Logistic Regression": lr, "Random Forest": rf, "XGBoost": xgb}
