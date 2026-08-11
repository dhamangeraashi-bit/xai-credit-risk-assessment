"""
config.py
=========
Single source of truth for column names, category codes, protected-attribute
definitions, and the random seed used throughout the pipeline.

ASSUMPTION (documented in docs/ASSUMPTIONS.md, item A1):
The paper does not publish an explicit random seed. RANDOM_SEED=42 is fixed
here so that every script in this repository is internally reproducible.
Re-running with a different seed will shift metrics slightly but not the
qualitative conclusions (this is discussed in docs/ASSUMPTIONS.md).
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw", "german.csv")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
TABLES_DIR = os.path.join(PROJECT_ROOT, "outputs", "tables")
LOGS_DIR = os.path.join(PROJECT_ROOT, "outputs", "logs")

for _d in (DATA_PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, LOGS_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.25  # 75/25 split, stratified (Section V-C of the paper)

# ---------------------------------------------------------------------------
# Column names for the raw UCI Statlog German Credit Data (german.csv),
# in the exact order documented by Hofmann (1994).
# ---------------------------------------------------------------------------
COLUMN_NAMES = [
    "checking_account_status",   # 1  qualitative
    "duration_months",           # 2  numerical
    "credit_history",            # 3  qualitative
    "purpose",                   # 4  qualitative
    "credit_amount",             # 5  numerical
    "savings_account",           # 6  qualitative
    "employment_since",          # 7  qualitative
    "installment_rate",          # 8  numerical
    "personal_status_sex",       # 9  qualitative
    "other_debtors",             # 10 qualitative
    "residence_since",           # 11 numerical
    "property",                  # 12 qualitative
    "age",                       # 13 numerical
    "other_installment_plans",   # 14 qualitative
    "housing",                   # 15 qualitative
    "existing_credits",          # 16 numerical
    "job",                       # 17 qualitative
    "num_dependents",            # 18 numerical
    "telephone",                 # 19 qualitative
    "foreign_worker",            # 20 qualitative
    "target_raw",                # 21 class: 1 = Good, 2 = Bad
]

CATEGORICAL_COLUMNS = [
    "checking_account_status", "credit_history", "purpose", "savings_account",
    "employment_since", "personal_status_sex", "other_debtors", "property",
    "other_installment_plans", "housing", "job", "telephone", "foreign_worker",
]

NUMERICAL_COLUMNS = [
    "duration_months", "credit_amount", "installment_rate",
    "residence_since", "age", "existing_credits", "num_dependents",
]

# ---------------------------------------------------------------------------
# Target definition
# ---------------------------------------------------------------------------
# Raw coding: 1 = Good credit risk, 2 = Bad credit risk.
#
# ASSUMPTION (docs/ASSUMPTIONS.md, item A3 -- IMPORTANT, please read):
# The paper reports very high recall (e.g. 0.937 for XGBoost) together with
# moderate precision (0.756) and reasonably high accuracy (0.744) -- a
# signature of "positive class = majority class" behaviour on a 700/300
# imbalanced set, NOT of "positive class = minority Bad class". We therefore
# adopt the target = 1 (Good), 0 (Bad) convention used in the large majority
# of public German-Credit tutorials/notebooks (i.e. df.target.replace({2: 0})
# leaving the original "1" label as-is). Empirically, this convention
# reproduces Table II to within roughly 2-6 percentage points on every cell
# using the hyperparameters in src/models.py -- see docs/ASSUMPTIONS.md for
# the full derivation and the alternative convention we ruled out.
#
# Practical consequence: model.predict_proba(x)[:, 1] is P(Good credit).
# "Probability of good creditworthiness" in Fig. 5's caption is exactly this
# quantity -- no 1-minus transform is needed anywhere downstream.
GOOD_CODE = 1
BAD_CODE = 2

# ---------------------------------------------------------------------------
# Protected attributes for fairness auditing (Section VII-B / Table III)
# ---------------------------------------------------------------------------
# Sex proxy, from attribute 9 (personal_status_sex):
#   A92 = female:divorced/separated/married, A95 = female:single -> "female"
#   A91, A93, A94 (all "male: ...")                              -> "male"
FEMALE_CODES = {"A92", "A95"}

# Age-group split used throughout the fairness literature for this dataset.
AGE_THRESHOLD = 25

# "No checking account" (A14) used as the thin-file / financially-excluded
# proxy in Section VII-C.
THIN_FILE_CODE = "A14"
