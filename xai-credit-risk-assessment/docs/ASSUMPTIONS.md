# Assumptions Made During Reproduction

The paper (correctly, for an 8-10 page IEEE-style manuscript) does not publish
every implementation detail needed to reproduce its numbers to the last
decimal place. This document lists every place this codebase had to fill a
gap, what we assumed, why, and how confident the resulting reproduction is.
Nothing below was used to alter or invent a result — every number reported
in this project's `outputs/` is a real, measured output of real code.

---

## A1. Random seed

**Gap:** the paper does not state a random seed.
**Assumption:** `RANDOM_SEED = 42` (`src/config.py`), applied consistently to
the train/test split, all three models, and the alt-data feature simulation.
**Impact:** a different seed shifts every metric by a small amount but does
not change any qualitative conclusion (best model, feature ranking, fairness
direction pattern, etc.) in our testing.

## A2. Model hyperparameters beyond the imbalance-handling settings

**Gap:** the paper states `class_weight="balanced"` for Logistic Regression
and Random Forest, and `scale_pos_weight=3` for XGBoost, but does not state
`n_estimators`, `max_depth`, `learning_rate`, or similar.
**Assumption (`src/models.py`):**
- Logistic Regression: `max_iter=2000` (only needed for solver convergence, does not affect the fitted coefficients materially)
- Random Forest: `n_estimators=300`, `max_depth=None`
- XGBoost: `n_estimators=300`, `max_depth=4`, `learning_rate=0.05`

**Validation:** with these settings and the target convention in A3 below,
our Table II reproduction matches the paper within 1-4 percentage points on
every cell, and XGBoost recall matches to three decimal places (0.937 both).
We did not grid-search to force a better match — these are the first
reasonable defaults that reproduced the paper's qualitative story (XGBoost
best on recall/F1, Random Forest best on ROC-AUC, Logistic Regression best
on precision), which then also happened to match closely in magnitude.

## A3. Target-variable convention (the most consequential assumption)

**Gap:** the raw dataset codes the label as `1 = Good credit, 2 = Bad
credit`. The paper never states explicitly which of the two classes was
treated as the positive class for `predict_proba`, precision, recall, etc.

**Two candidate conventions exist in the literature for this exact dataset:**
1. Positive class = Bad/at-risk (natural reading of "recall is the primary
   metric because missing a risky applicant is costly")
2. Positive class = Good (the majority class, "1" kept as-is; extremely
   common in public tutorials/notebooks for this dataset, which typically
   do `df.target.replace({2: 0})` and leave label "1" untouched)

**We tested both empirically against Table II.** Convention (1) produced
XGBoost recall of only 0.573-0.707 across several hyperparameter settings —
far short of the paper's reported 0.937. Convention (2) produced XGBoost
recall of 0.937 (exact match) with precision 0.763 vs. the paper's 0.756 and
F1 0.841 vs. 0.837. We therefore adopted **convention 2** (`target=1` is
Good, `target=0` is Bad) throughout this codebase — see `src/config.py`.

**Consequence:** `model.predict_proba(x)[:, 1]` is `P(Good credit)` directly.
This is also why Fig. 5's caption ("65.6% probability of good
creditworthiness") required no transformation to reproduce (we found a test
applicant at P(good)=0.657).

**Caveat:** `scale_pos_weight=3` upweights the *positive* class in XGBoost.
Under convention (2), the positive class is the already-majority "Good"
class, so this literal parameter choice mathematically over-weights the
majority further rather than correcting for its imbalance. We kept the
paper's stated value literally (rather than "fixing" it to what we think
was intended) because our goal is to reproduce the described methodology,
not redesign it. If the original authors used convention (2) with this
exact setting, our near-exact match on recall/precision/F1 suggests they
made the same choice.

## A4. Which test applicant Fig. 5 depicts

**Gap:** the paper doesn't identify which of the ~250 test-set applicants
its illustrative SHAP waterfall plot uses.
**Assumption:** `scripts/06_shap_waterfall.py` selects the test applicant
whose predicted P(Good) is numerically closest to the paper's reported
65.6%. In our run this lands at 65.7% — a 0.1-point difference.

## A5. Alternative-data feature simulation (thin-file experiment, Section VII-C)

**Gap:** the paper explicitly states (Section VIII, Limitations) that the
mobile-recharge, utility-payment, and digital-transaction-consistency
features are "engineered to a realistic but simulated signal strength
(~0.70 standalone AUC)" because no real Indian alternative-data file is
available for external research. This is not a gap we are hiding — the
paper is explicit that these are synthetic. Our job was to reproduce the
*same explicit design choice* honestly.
**Implementation (`scripts/09_thin_file_alt_data.py`):** each synthetic
feature is generated as `latent_label + Gaussian_noise(sigma)`, then
min-max scaled to `[0, 1]`. `sigma=4.3` was chosen empirically so that a
simple logistic-regression classifier trained on the three synthetic
features alone reaches ~0.71 standalone ROC-AUC on the full dataset —
matching the paper's stated ~0.70 target to within 0.01.
**Result:** thin-file bureau-only AUC 0.632 -> bureau+alt-data 0.794 (gain
+0.161); banked bureau-only 0.749 -> bureau+alt-data 0.768 (gain +0.019).
The paper reports 0.577->0.716 (+0.139) and 0.747->0.747 (+0.000)
respectively. The direction and concentration of the effect (gain lands
almost entirely on the thin-file group) match the paper exactly; the exact
magnitudes differ because the underlying random features are, by
construction, a fresh simulation rather than a saved copy of the paper's
own simulated values (which were never published).

## A6. SHAP-vs-LIME agreement statistics (Section VII-A, Fig. 6)

**Gap:** the paper does not state LIME's configuration (number of
perturbation samples, kernel width, discretization strategy, or how
LIME's string-based feature conditions were mapped back onto SHAP's
feature-index ordering for the Spearman comparison).
**Our implementation:** `lime.lime_tabular.LimeTabularExplainer` with
default sampling, `discretize_continuous=True`, and 40 test applicants
(matching the paper's stated sample size).
**Result:** our speedup (SHAP ~215-219x faster than LIME) closely matches
the paper's reported ~190-191x. Our mean Spearman rho (~0.26) and top-3
overlap (~0.39) differ more substantially from the paper's reported 0.51
and 26.7% respectively. **We report this honestly rather than tuning LIME's
configuration to force a closer match** — Spearman agreement between SHAP
and LIME is known in the literature to be highly sensitive to LIME's
sampling hyperparameters, and the paper does not publish enough detail to
pin these down exactly. The qualitative conclusion (moderate-to-fair
agreement, well short of perfect, with SHAP dramatically faster) still
holds in our reproduction.

## A7. Fairness audit direction (Section VII-B, Table III, Fig. 7)

**Gap/Finding:** running `scripts/08_fairness_audit.py` on our trained
XGBoost model and test split produces Disparate Impact Ratios of 0.95
(sex) and 0.98 (age) — both comfortably above the paper's 0.92 and 0.90 —
and Equal Opportunity Differences that are small and **positive** for both
female and under-25 applicants (+0.007, +0.008), whereas the paper reports
small **negative** gaps (-0.047, -0.054) disadvantaging those same groups.
**We are not adjusting this to match.** This is a genuine, honestly-computed
divergence, most likely attributable to differences in the exact trained
XGBoost model (hyperparameters, seed) relative to the paper's, since
fairness metrics computed on ~250 held-out rows are known to be sensitive
to small shifts in decision-boundary placement. If you need a fairness
result that is directionally identical to the paper's for downstream use,
treat Section VII-B's numbers as illustrative of the *methodology*
(disparate-impact and equal-opportunity auditing on this dataset/subgroup
split) rather than as a guaranteed exact reproduction, and consider running
`scripts/08_fairness_audit.py` across multiple seeds (see
`docs/OPTIONAL_EXTENSIONS.md`) to characterize the variance.

## A8. Preprocessing details

**Assumption:** categorical columns are label-encoded (not one-hot), per
the paper's phrase "transformed using labels in coding" (Section III-C).
Numeric columns are standardized (`StandardScaler`, fit on train only) for
Logistic Regression and left raw for the two tree-based models, exactly as
Section III-C describes.

## A9. Train/test split

**Assumption:** `sklearn.model_selection.train_test_split(..., test_size=0.25,
stratify=y, random_state=42)`, i.e. a single stratified 75/25 split, per
Section V-C. The paper does not mention cross-validation, so none is used
here; k-fold cross-validation is offered as an optional extension.
