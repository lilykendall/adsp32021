# ADSP 31021 — Assignment 4: Model Monitoring

**Lily Kendall**

**Repo Link:** https://github.com/lilykendall/adsp32021

*I used Claude to assist with the assignment*

A model-monitoring workflow over the county-level cancer mortality dataset
(`cancer_reg.csv`). A **Linear Regression** model predicts `TARGET_deathRate`
from county demographics; **Evidently AI** then monitors input drift and
prediction drift as three controlled perturbations are applied to the held-out
test set.

**Task:** regression — predict `TARGET_deathRate` (mean per-capita cancer
mortalities per 100,000) from county-level census and cancer-registry features.

### Headline finding

The three scenarios produce three *different* monitoring failure modes, and no
single check catches all of them:

| Scenario | Accuracy damage | Prediction drift | Input drift | What would have caught it |
|---|---|---|---|---|
| **A** | mild (R² 0.496 → 0.455) | **detected** | **detected** | any of the three checks |
| **A+B** | **none** (R² 0.495) | **NOT detected** (p = 0.999) | **detected** | **input drift only** |
| **A+B+C** | severe (R² → 0.193) | **detected** | **detected** | any of the three checks |

Scenario A+B is the case that matters. The income and poverty shifts almost
exactly cancel in the linear model (−5.69 and +5.16 deaths/100k), so the model
looks **completely healthy** on accuracy *and* on prediction drift — 33 of 35
Evidently tests pass, every regression test among them — while a third of the
input rows contain negative median incomes. Only per-column input drift and the
pre-inference validation gate fire. That is the operational argument of this
whole assignment: **output monitoring alone is not sufficient.**

---

## Setup & execution

```bash
cd assignment4
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python run_pipeline.py          # full workflow, ~24s, no manual intervention
```

Each stage is also runnable on its own:

| Stage | Command | Writes |
|---|---|---|
| 1. Validate raw data | `python -m src.data` | `artifacts/metrics/data_validation.json` |
| 2. Clean + split | `python -m src.preprocess` | `data/train.csv`, `data/test_original.csv` |
| 3. Train model | `python -m src.model` | `artifacts/model.joblib`, `linear_coefficients.csv` |
| 4. Build scenarios | `python -m src.scenarios` | `data/test_scenario_*.csv` |
| 5. Input validation gate | `python -m src.validation` | `artifacts/metrics/input_validation.*` |
| 6. Score all datasets | `python -m src.evaluate` | `artifacts/metrics/`, `artifacts/figures/` |
| 7. Model comparison | `python -m src.compare` | `artifacts/metrics/model_comparison.*` |
| 8. Evidently monitoring | `python -m src.monitoring` | `artifacts/reports/*.html` |

**Reproducibility:** `random_state=42` for the split and for every estimator;
all tunable decisions are centralised in [`src/config.py`](src/config.py).

> **Note on the Evidently pin.** `requirements.txt` pins `evidently==0.4.40`,
> which uses the `Report` / `TestSuite` / `ColumnMapping` API. Evidently 0.7.x
> replaced this with a `Dataset` / `DataDefinition` API and removed
> `evidently.report`, so `src/monitoring.py` will not run on ≥ 0.7.

---

## 1. Dataset loading and data understanding

- **Source:** `cancer_reg.csv` — 3,047 US counties × 34 columns. The file is
  **latin-1**, not UTF-8 (county names contain `ñ`); `src/config.py` sets the
  encoding explicitly.
- **Target:** `TARGET_deathRate` — mean per-capita (100,000) cancer mortalities.
- **Objective:** supervised regression. Given a county's demographic,
  socioeconomic, education, insurance-coverage and cancer-incidence profile,
  predict its cancer death rate. The practical use would be prioritising
  counties for screening or oncology-resource investment.

**Key columns used in modelling** (full definitions in
[`data_dictionary.md`](data_dictionary.md)). The "effect" column is the fitted
per-raw-unit coefficient from §3 — worth reading alongside the meaning, because
several are counter-intuitive:

| Column | Meaning | Why it matters here |
|---|---|---|
| `incidenceRate` | Cancer diagnoses per 100,000 | **The dominant predictor** (‖std coef‖ = 10.75). Counties that diagnose more cancer bury more cancer patients — the single most direct causal path in the feature set. |
| `PctBachDeg25_Over` | % adults 25+ with a bachelor's | Second strongest (−7.49). Education proxies for income, health literacy, insurance and screening access all at once. |
| `PctPrivateCoverage` | % with private health coverage | −7.03. Insurance status governs screening and treatment access; the strongest of the four coverage columns. |
| `PctMarriedHouseholds` / `PercentMarried` | % married households / residents | −6.53 / +6.18. Large, **opposite-signed**, and highly collinear — a classic suppression pair. Neither coefficient should be read causally in isolation. |
| `medIncome` | Median county income | Perturbed in **Scenario A**. Coefficient is *positive* (+0.000142/$) once poverty, education and coverage are controlled for — see the collinearity note below. |
| `povertyPercent` | % of populace in poverty | Perturbed in **Scenario B**. +0.258 per point. |
| `AvgHouseholdSize` | Mean household size | Perturbed in **Scenario C**. −7.46 per person — the largest per-raw-unit coefficient of the three, which is why Scenario C does the most damage. |
| `avgAnnCount` | Mean annual cancer diagnoses | Kept; see the leakage discussion below. |
| `popEst2015` | County population | Scale control — lets count-valued columns be interpreted per-capita. |

**Data validation findings** (from `artifacts/metrics/data_validation.json`):

| Check | Result |
|---|---|
| Rows × columns | 3,047 × 34 |
| Duplicate rows | 0 |
| Categorical columns | `binnedInc`, `Geography` |
| Constant columns | none |
| `PctSomeCol18_24` missing | 2,285 / 3,047 (**75.0%**) |
| `PctPrivateCoverageAlone` missing | 609 / 3,047 (20.0%) |
| `PctEmployed16_Over` missing | 152 / 3,047 (5.0%) |
| Negative values anywhere | none |
| Percentage columns outside [0, 100] | none |
| `MedianAge` > 100 | 30 counties — max **624**, clearly a scaling error |
| `AvgHouseholdSize` < 1 | 61 counties — min **0.0221**, recorded as a fraction |
| `studyPerCap` == 0 | 1,931 counties (63.4%) — genuine zero, not missingness |

The last row is worth flagging: most counties host no cancer clinical trials at
all, so `studyPerCap` is zero-inflated by nature. It is a real measurement, not a
defect, and is left untouched.

**Columns excluded from the model, and why:**

| Column | Reason |
|---|---|
| `Geography` | Free-text county identifier, 3,047 unique values — no generalisable signal, and one-hot encoding it would give every county its own parameter |
| `binnedInc` | String decile of `medIncome`; redundant, and keeping it would *mask* the Scenario A income shift — the bin label is a fixed string that would not move with the perturbed value, so drift detection would see a column that changed and a column that didn't, from the same underlying quantity |
| `PctSomeCol18_24` | 75% missing — imputation would fabricate three quarters of the column |
| `PctPrivateCoverageAlone` | 20% missing *and* near-collinear with `PctPrivateCoverage`, which is complete. Dropping it costs almost nothing and removes an unstable coefficient from a linear model |
| `avgDeathsPerYear` | **Target leakage** (see below) |

This leaves **28 features** (34 raw − 1 target − 5 excluded).

**The leakage call on `avgDeathsPerYear` vs `avgAnnCount`.** These two are not
equivalent and are treated differently.

`avgDeathsPerYear` is **excluded**. The target is a *rate* of cancer deaths;
`avgDeathsPerYear` is the *count* of cancer deaths, i.e. the numerator of the
target. `popEst2015` — effectively the denominator — is in the feature set, so a
model given both can reconstruct `TARGET_deathRate` almost arithmetically. That
is not prediction, it is division, and it would produce an impressive R² that
collapses the moment the model met a county whose death count was not yet known.
Since the entire point of the model is to predict mortality *before* it is
observed, a feature that presupposes the mortality count is unusable.

`avgAnnCount` is **kept**. It counts cancer *diagnoses*, not deaths. A diagnosis
is observed years before the mortality it may eventually contribute to, so it is
genuinely available at prediction time. It is certainly correlated with the
target, but correlation with the target is what a predictor *is* — the leakage
test is whether the feature is available at prediction time, and this one is.
`incidenceRate` (the per-capita version of the same quantity) is kept for the
same reason and is the model's strongest predictor.

**A note on the `medIncome` coefficient.** The fitted effect of income is
*positive* (+0.000142 deaths/100k per dollar) — meaning that, holding poverty,
education, employment and insurance coverage constant, higher income is
associated with slightly *higher* predicted mortality. This is not a plausible
causal statement. It is collinearity: `medIncome`, `povertyPercent`,
`PctBachDeg25_Over` and the coverage columns all measure overlapping aspects of
county affluence, and OLS distributes a shared effect across correlated
predictors more or less arbitrarily. The consequence for this assignment is
concrete rather than academic — it is *why* Scenario A moves predictions
**down** instead of up, and why A and B nearly cancel in §7. It is also a
standing argument for not interpreting any single coefficient here causally.

**Assumptions.**

1. **Rows are independent observations.** Counties are treated as independent
   draws. They are not — neighbouring counties share health systems, industries
   and demographics, so residuals are almost certainly spatially autocorrelated.
   This does not bias the point predictions much, but it means any standard
   error or confidence interval computed from this model would be too narrow.
2. **Temporal alignment.** The mortality figures cover 2010–2016 while the census
   estimates are from 2013. Features are assumed contemporaneous with the
   outcome window. A county whose economy changed sharply within that span is
   mis-measured.
3. **Percentage columns are on a 0–100 scale**, verified — no percentage column
   in the raw file falls outside [0, 100].
4. **Missingness is ignorable** conditional on the observed features, which is
   what median imputation assumes. This is the weakest assumption in the
   pipeline: `PctEmployed16_Over` is 5% missing and that missingness is very
   unlikely to be random across county types.
5. **Ecological data supports only ecological conclusions.** Every relationship
   here holds between *counties*, not between people. Inferring anything about
   an individual from these coefficients is the ecological fallacy.

---

## 2. Preprocessing and train/test split

| Decision | Value |
|---|---|
| Split ratio | 80 / 20 |
| Random seed | 42 |
| Train / test rows | 2,437 / 610 |
| Target | `TARGET_deathRate` |
| Features | 28 numeric columns |
| Missing-value handling | Median imputation, **fit on training rows only** |
| Scaling | `StandardScaler`, **fit on training rows only** |
| Categorical handling | Both categoricals excluded (see §1); no encoding needed |

**Leakage control.** Only *stateless* operations run before the split
([`src/preprocess.py::clean`](src/preprocess.py) — dropping columns, nulling
impossible values). Every step that learns a statistic — the imputer's medians
and the scaler's means and standard deviations — lives inside the sklearn
`Pipeline` in [`src/model.py`](src/model.py), so it is fit on the 2,437 training
rows and merely *applied* to the test and scenario sets. Because the fitted
pipeline is serialised whole into `artifacts/model.joblib`, the scenario
datasets are transformed with training-set statistics too, exactly as a
production model would.

**Why scale at all for a linear model?** OLS does not require it — coefficients
and predictions are identical either way. It is done so the coefficients are
directly comparable across columns measured in dollars, percentages and people,
which is what makes the coefficient table in §3 readable. The per-raw-unit
coefficients are recovered by dividing by the scaler's per-column scale.

**Invalid-value treatment.** `MedianAge > 100` (30 rows) and
`AvgHouseholdSize < 1` (61 rows) are set to null and then median-imputed rather
than dropped.

The alternative — dropping the ~91 affected rows — is defensible, and was
rejected for two reasons. First, each of those rows contains one corrupt cell
and 27 sound ones; dropping the row to escape the bad cell discards the good
data with it. Second, and more importantly, the corruption is **not missing at
random**: both defects cluster in small rural counties, which are also the
counties with the most extreme death rates. Dropping them would systematically
thin the tail of the training distribution and bias the model toward large
suburban counties. Nulling confines the damage to the untrustworthy cell and
keeps all 3,047 counties.

This choice has a visible downstream consequence in Scenario C, noted in §7:
the 13 test rows whose `AvgHouseholdSize` is null do not respond to the +2
perturbation at all, because `NaN + 2` is still `NaN` and still gets imputed to
the training median.

---

## 3. Baseline model development

| Setting | Value |
|---|---|
| **Model** | **`LinearRegression`** (ordinary least squares) |
| Feature set | 28 numeric columns (see §1) |
| Preprocessing | Median imputation → `StandardScaler`, both inside the pipeline |
| Regularisation | None |
| `random_state` | 42 (split; OLS itself is deterministic) |
| Training rows | 2,437 |
| Saved to | `artifacts/model.joblib`, reproducible via `python -m src.model` |

**Why Linear Regression.** The choice is deliberate and costs some accuracy.

1. **Interpretability that the monitoring analysis actually uses.** Every
   coefficient is an explicit statement about how the model responds to an
   input. That makes the scenario effects *predictable in advance* rather than
   merely observable afterwards: multiplying the per-raw-unit coefficient by
   each perturbation's delta gives the exact per-row prediction shift each
   scenario will cause, before a single prediction is scored. §7 confirms these
   predictions to the decimal. No tree ensemble supports that.
2. **It extrapolates, which is the honest behaviour here.** A tree ensemble
   cannot produce a prediction outside the range of its training leaves — given
   a `medIncome` of −15,965 it returns whatever it learned for the poorest
   county it ever saw, silently. A linear model keeps extending the fitted
   relationship. Neither is *correct* on impossible input, but the linear model
   at least fails loudly enough to be noticed.
3. **The relationships are largely monotone.** `incidenceRate`, education and
   coverage all trend smoothly against mortality; the ceiling for a nonlinear
   model on this dataset turns out to be modest (0.542 vs 0.496 R²).

**Fitted coefficients** (`artifacts/metrics/linear_coefficients.csv`).
`std_coef` is the effect of a one-standard-deviation move — comparable across
columns; `raw_unit_coef` is the effect of one raw unit. Intercept: **178.609**.

| Feature | std coef | per raw unit |
|---|---|---|
| `incidenceRate` | **+10.752** | +0.1956 per diagnosis/100k |
| `PctBachDeg25_Over` | **−7.491** | −1.424 per point |
| `PctPrivateCoverage` | −7.032 | −0.666 per point |
| `PctMarriedHouseholds` | −6.527 | −0.999 per point |
| `PercentMarried` | +6.184 | +0.897 per point |
| `PctEmpPrivCoverage` | +3.555 | +0.377 per point |
| `PctEmployed16_Over` | −3.492 | −0.435 per point |
| `MedianAgeMale` | −3.465 | −0.657 per year |
| … | | |
| `AvgHouseholdSize` | −1.866 | **−7.459 per person** |
| `medIncome` | +1.683 | **+0.000142 per dollar** |
| `povertyPercent` | +1.638 | **+0.258 per point** |

The three perturbed columns are all *mid-strength* predictors by standardised
coefficient — none is in the top eight. That is worth holding onto when reading
§7: the scenarios do not attack the model's most important inputs, and A+B+C
still cuts R² by 61%.

**Model comparison** (`artifacts/metrics/model_comparison.csv`, produced by
[`src/compare.py`](src/compare.py)). A Random Forest was trained on the identical
pipeline and split as a benchmark. It is **not** the deployed model and is never
monitored — it exists to quantify what the linear assumption costs and to make
the extrapolation argument in §7 concrete.

| Model | RMSE | MAE | R² (test) | R² (5-fold CV on train) |
|---|---|---|---|---|
| **LinearRegression** | 20.31 | 15.11 | **0.4957** | 0.5020 |
| RandomForest (300 trees, `min_samples_leaf=2`) | 19.37 | 14.20 | 0.5416 | 0.5222 |

The forest is better by **0.046 R² / 0.95 RMSE** — real but modest, and the CV
figures confirm the gap is genuine rather than a quirk of this particular test
split. A `RidgeCV` sweep over 25 alphas was also run during model selection and
scored R² = 0.4977, i.e. indistinguishable from plain OLS; regularisation buys
nothing here, so it was not kept. The ~0.05 R² was traded for interpretability
and for the far more informative failure behaviour documented in §7.

---

## 4. Baseline model evaluation

Evaluated on the untouched original test set (610 counties):

| Metric | Value |
|---|---|
| RMSE | **20.31** |
| MAE | **15.11** |
| R² | **0.496** |
| Mean prediction | 178.77 (actual mean 178.89) |
| Prediction std | 21.27 |
| Prediction range | 115.60 – 247.25 |

Diagnostics: `artifacts/figures/diagnostics_original.png` (predicted-vs-actual
and residual plots).

**Interpretation.** R² ≈ 0.50 means the model explains about half the variance in
county death rates. For an ecological model built from census aggregates that is
a reasonable result and not an embarrassing one, but it is nowhere near good
enough to drive a resource-allocation decision on its own. An MAE of 15.1
against a mean death rate of 178.9 is an average error of **8.4%**; a county
predicted at 180 could plausibly be anywhere from 165 to 195.

The mean prediction (178.77) tracks the actual mean (178.89) almost exactly, so
the model is unbiased *on average* — but the prediction standard deviation
(21.27) is compressed relative to the spread of the actuals. This is the
signature regression-to-the-mean behaviour of a least-squares fit with moderate
R²: the model systematically **under-predicts the highest-mortality counties and
over-predicts the lowest**. Operationally that is the worst possible direction of
error for this use case, because the counties the model shrinks toward the mean
are exactly the ones that most need intervention.

**Limitations.**

- **Ecological inference.** Every relationship holds between counties, not
  people. The model cannot say anything about an individual's cancer risk.
- **Correlation, not causation.** The `medIncome` coefficient (§1) is positive,
  which is a collinearity artefact rather than a claim that raising incomes
  raises mortality. No coefficient here should be read as a policy lever.
- **Confidence is unquantified.** Point predictions ship with no interval. OLS
  intervals could be computed, but the independence assumption they rest on
  (§1) is violated by spatial autocorrelation, so they would be too narrow.
- **Half the variance is unexplained**, and the missing drivers are plausibly
  things absent from the dataset entirely: smoking prevalence, occupational
  exposure, distance to the nearest oncology centre.
- **Fixed feature ranges.** The model has only ever seen incomes from \$22,640 to
  \$125,635. Nothing in the estimator itself objects to being asked about a
  county outside that range — which is precisely what §5 and §6 exploit.

---

## 5. Evidently AI monitoring setup

**Reference dataset:** the **scored original test set** — features + actual
`TARGET_deathRate` + model `prediction` (`data/scored_original.csv`). Using the
untouched test set as reference rather than the training set means any drift
Evidently reports is attributable to the controlled perturbation alone, and not
to the ordinary sampling difference between a training and a holdout sample.
Checked: running the original test set against itself yields **0 drifted
columns of 30** and 0 failed tests, which confirms the reference is clean and
that every later signal is caused by a perturbation.

**Current dataset:** each scored scenario test set, in turn.

**Checks configured** ([`src/monitoring.py`](src/monitoring.py)):

| Check | What it catches | Why it is relevant |
|---|---|---|
| `DataDriftPreset` | Per-column input distribution shift (K–S test per numeric column) | Detects the injected changes to `medIncome`, `povertyPercent`, `AvgHouseholdSize`. **The only check that fired on all three scenarios.** |
| `TargetDriftPreset` | Drift in the target **and** in `prediction` | This is the output/prediction drift signal — the one available in production before labels arrive |
| `DataQualityPreset` | Per-column min/max/mean summaries and value ranges | Where the *impossible* values become visible: it is what surfaces a `medIncome` minimum of −15,965 rather than merely reporting that the column moved |
| `RegressionPreset` | RMSE / MAE / error distribution, reference vs current | Quantifies the accuracy damage — but requires ground truth, so in production it is retrospective |
| `DataDriftTestPreset` + `RegressionTestPreset` (`TestSuite`) | The same checks as pass/fail assertions | What a scheduled production monitoring job would actually alert on |

Two artefacts are written per dataset: `report_<name>.html` (visual) and
`tests_<name>.html` (35 pass/fail assertions), in `artifacts/reports/`.

**Why this particular mix.** The presets are chosen around *when each signal
becomes available*, which is the constraint that matters in production and is
invisible in a classroom setting where labels are always to hand:

- Inputs arrive **immediately**. Input drift and data-quality checks can fire
  the moment a batch is submitted, before a single prediction is made.
- Predictions arrive **immediately after scoring**. Prediction drift needs no
  labels at all, which makes it the workhorse of unsupervised monitoring.
- Ground-truth death rates arrive **years later**, through registry and census
  cycles. Regression performance is therefore a retrospective confirmation, not
  an alerting mechanism. By the time RMSE moves on a dashboard, every decision
  made on the model's output has already been made.

The results in §8 justify keeping all three layers rather than just the cheapest
one — Scenario A+B is caught by the input layer and by nothing else.

**Beyond Evidently: the pre-inference validation gate.** Drift detection answers
"have these inputs changed relative to a reference?" It cannot answer "are these
inputs *possible*?" — a batch of uniformly negative incomes would be perfectly
consistent with itself and would raise no alarm at all if it became the new
normal. [`src/validation.py`](src/validation.py) adds that missing check,
comparing every column against two separate standards:

| Check | Meaning | Production action |
|---|---|---|
| **Hard bounds** (`config.INPUT_BOUNDS`) | The value is physically impossible — a negative median income, a percentage above 100 | **REJECT** the row |
| **Training-range excursion** | The value is possible but was never seen in training, so the model is extrapolating | **WARN** — flag the prediction as low-confidence |

Bounds are explicit for the 11 columns with meaningful physical limits, and
inferred as [0, 100] for anything matching the `Pct*` / `Percent*` naming
convention. The out-of-distribution check tolerates up to
`OOD_ALERT_THRESHOLD` = 5% of a batch before warning, since a handful of test
rows always fall marginally outside the training range through ordinary sampling
variation.

**The gate runs in report-only mode.** It records every violation and writes the
evidence to `artifacts/metrics/input_validation.json` but deliberately drops
nothing, for two reasons: the assignment requires the perturbed values be scored
exactly as specified so Evidently sees the true shift, and dropping rows would
leave the four datasets with different row counts and non-comparable metrics.
The blocking behaviour is nonetheless implemented — `apply_gate(..., enforce=True)`
on Scenario A+B+C scores 419 of 610 rows and rejects 191 — so the safeguard is
demonstrable rather than merely described.

---

## 6. Modified test dataset creation

Perturbations are applied **cumulatively** to a copy of the original test set;
`data/test_original.csv` is never overwritten.

| Dataset | File | Change |
|---|---|---|
| Original | `data/test_original.csv` | — |
| Scenario A | `data/test_scenario_a.csv` | `medIncome` − 40,000 |
| Scenario A+B | `data/test_scenario_ab.csv` | A, plus `povertyPercent` + 20 |
| Scenario A+B+C | `data/test_scenario_abc.csv` | A + B, plus `AvgHouseholdSize` + 2 |

> **Column-name note.** The assignment writes these lowercased
> (`medianincome`, `povertypercent`, `avghouseholdsize`); the actual CSV headers
> are `medIncome`, `povertyPercent`, `AvgHouseholdSize`.

**Validation** (`artifacts/metrics/scenario_verification.json`) asserts, per
scenario, that (a) each targeted column moved by *exactly* the specified delta
for every row, (b) no other column changed, and (c) the row count is unchanged.
**All checks pass for all three scenarios.**

| Column | Original mean | Modified mean | Modified range |
|---|---|---|---|
| `medIncome` (A) | 46,989 | 6,989 | **−15,965** to 68,477 |
| `povertyPercent` (B) | 17.03 | 37.03 | 23.70 to 60.60 |
| `AvgHouseholdSize` (C) | 2.52 | 4.52 | 3.91 to 5.97 |

**The out-of-range side effect, and why it matters.** The perturbations push
values outside their natural ranges, but — importantly — *not uniformly*. The
gate results (`artifacts/metrics/input_validation.csv`) separate two very
different problems:

| Column | Verdict | Detail |
|---|---|---|
| `medIncome` | **Impossible** | **191 of 610 rows (31.3%) are negative.** 553 rows (90.7%) fall outside the training range [22,640 – 125,635] |
| `povertyPercent` | Legal but extreme | Max 60.6% is a valid percentage. 47 rows (7.7%) exceed the training maximum of 47.4% |
| `AvgHouseholdSize` | Physically plausible, statistically absurd | 5.97 people per household is possible somewhere on earth. But **596 of 610 rows (97.7%)** fall outside the training range [1.86 – 3.97] |

So only Scenario A produces genuinely *invalid* data. Scenarios B and C produce
data that is merely **unprecedented** — every value could exist, but the model
has never seen anything like it. The gate status is `REJECT` for all three
scenarios (driven entirely by `medIncome`) and `PASS` for the original test set.

This distinction is the practical point. A schema or bounds check catches
Scenario A instantly and would have prevented the model from ever scoring it. It
catches **neither B nor C** — those need distribution-aware monitoring, because
there is nothing wrong with any individual value. And Scenario C, the one that
does the most damage to accuracy (§7), is the one a bounds check is least
equipped to see. Both layers are necessary; neither is sufficient.

---

## 7. Scenario-based model scoring and accuracy verification

The single trained Linear Regression model scored against all four datasets
(`artifacts/metrics/scenario_metrics.csv`):

| Dataset | n | RMSE | MAE | R² | Pred. mean | Pred. std | Pred. min | Pred. max |
|---|---|---|---|---|---|---|---|---|
| Original | 610 | 20.31 | 15.11 | **0.496** | 178.77 | 21.27 | 115.60 | 247.25 |
| Scenario A | 610 | 21.13 | 15.92 | **0.455** | 173.09 | 21.27 | 109.91 | 241.56 |
| Scenario A+B | 610 | 20.32 | 15.13 | **0.495** | 178.25 | 21.27 | 115.07 | 246.72 |
| Scenario A+B+C | 610 | 25.69 | 20.42 | **0.193** | 163.65 | 21.51 | 100.15 | 231.80 |

Prediction distributions overlaid: `artifacts/figures/prediction_distributions.png`.
Model-comparison degradation paths: `artifacts/figures/model_comparison_r2.png`.

### The damage was predictable from the coefficients

Because the model is linear and each perturbation is a uniform additive shift,
the effect on every prediction is exactly `raw_unit_coef × delta` — computable
before scoring anything:

| Scenario | Column | Coefficient | Delta | Predicted shift | Observed shift in mean prediction |
|---|---|---|---|---|---|
| A | `medIncome` | +0.000142 | −40,000 | **−5.69** | 178.77 → 173.09 = **−5.69** ✓ |
| B | `povertyPercent` | +0.258 | +20 | **+5.16** | 173.09 → 178.25 = **+5.16** ✓ |
| C | `AvgHouseholdSize` | −7.459 | +2 | **−14.92** | 178.25 → 163.65 = **−14.60** ✓* |

*\* Scenario C lands 0.32 short of the prediction because 13 of the 610 test rows
have a null `AvgHouseholdSize` (§2). `NaN + 2` is still `NaN`, so those rows are
imputed to the training median and do not shift at all. Confirming this: it is
also why Scenario C is the only scenario where the prediction standard deviation
changes (21.27 → 21.51) — a uniform shift moves every prediction equally and
leaves the spread untouched, but 13 rows failing to move introduces genuine
extra variance. A small detail that quietly demonstrates the imputation is
behaving exactly as designed.*

### Reading the degradation path

**Scenario A degrades accuracy mildly and in a counter-intuitive direction.**
R² falls 0.496 → 0.455 and every prediction drops by 5.69. Predictions moving
*down* when incomes collapse is the collinearity artefact from §1 surfacing:
holding poverty and education fixed, the fitted `medIncome` coefficient is
positive. The model is not wrong about the world so much as being asked a
question its coefficients were never identified to answer.

**Scenario A+B is the assignment's most important result: the damage cancels.**
R² returns to 0.495 — statistically indistinguishable from the 0.496 baseline —
and RMSE returns to 20.32 against a baseline of 20.31. The income shift subtracts
5.69 and the poverty shift adds 5.16; the residual net movement in the mean
prediction is **0.53 deaths per 100,000**, a rounding error.

By every accuracy metric available, the model on Scenario A+B is **as healthy as
it was on clean data**. A third of its input rows contain negative median
incomes. Any monitoring strategy resting on accuracy alone — or, as §8 shows, on
prediction drift alone — reports that this model is fine.

**Scenario A+B+C is a genuine collapse.** R² falls to 0.193, RMSE rises 26% to
25.69, MAE rises 35%. `AvgHouseholdSize` carries by far the largest per-raw-unit
coefficient of the three perturbed columns (−7.46 per person, against −0.000142
per dollar and +0.258 per point), so a +2 shift moves every prediction by −14.92
— nearly triple the size of either previous effect and about 70% of a full
prediction standard deviation. Nothing cancels it, and the mean prediction ends
**15.24 below** the actual mean of 178.89.

### The comparison that justifies the model choice

The same three scenarios, scored with the Random Forest benchmark:

| Model | Original | Scenario A | A+B | A+B+C |
|---|---|---|---|---|
| **LinearRegression** (deployed) | 0.496 | 0.455 | **0.495** | **0.193** |
| RandomForest (benchmark) | 0.542 | 0.459 | 0.400 | 0.437 |
| **Mean prediction — linear** | 178.77 | 173.09 | 178.25 | **163.65** |
| **Mean prediction — forest** | 179.50 | 185.11 | 187.60 | 184.93 |

The two model families fail in genuinely opposite ways, and neither is safe:

- **The forest cannot extrapolate.** A `medIncome` of −15,965 lands in the same
  terminal leaf as the poorest county in the training data, so the forest's
  answer is capped at "what I learned about poor counties." Its R² never falls
  below 0.40 — it *degrades quietly* under any input, however impossible, and
  even partially recovers at A+B+C. On a dashboard tracking RMSE, the forest's
  worst scenario is a 14% degradation you might not look twice at.
- **The linear model extrapolates without bound.** It applies the fitted
  relationship to inputs no county has ever exhibited and, at A+B+C, produces a
  61% collapse in R² that no operator could miss.
- **They even move in opposite directions.** The forest's mean prediction rises
  in every scenario (179.5 → 185–188); the linear model's falls (178.8 → 163.6).
  Two defensible models trained on identical data disagree about the *sign* of
  the effect on impossible input — which is the clearest possible warning that
  neither prediction means anything once the inputs leave the training range.

The operational conclusion is not that one model is right. It is that
**out-of-range behaviour is arbitrary**, and that arbitrariness has to be handled
by a validation gate upstream of the model rather than by hoping the estimator
degrades gracefully.

---

## 8. Monitoring analysis for input and output changes

Per-column drift detection (`artifacts/metrics/drift_by_column.csv`), K–S test,
threshold p < 0.05:

| Dataset | `medIncome` | `povertyPercent` | `AvgHouseholdSize` | `prediction` | Total drifted / 30 | Failed tests / 35 |
|---|---|---|---|---|---|---|
| Original | ✗ (p = 1.00) | ✗ (p = 1.00) | ✗ (p = 1.00) | ✗ (p = 1.00) | **0** | **0** |
| Scenario A | **✓** (p ≈ 5e−278) | ✗ (p = 1.00) | ✗ (p = 1.00) | **✓** (p = 2.4e−5) | **2** | **3** |
| Scenario A+B | **✓** (p ≈ 5e−278) | **✓** (p ≈ 4e−257) | ✗ (p = 1.00) | **✗ (p = 0.999)** | **2** | **2** |
| Scenario A+B+C | **✓** | **✓** | **✓** (p = 0.0) | **✓** (p = 8e−29) | **4** | **8** |

`TARGET_deathRate` never drifted (p = 1.00 in every scenario), which is correct —
the actuals were never modified.

### Input drift: every injected change detected, and nothing else

Evidently caught **all three** perturbed columns, at p-values low enough to be
reported as zero, and produced **no false positives** across the 25 untouched
feature columns (or the target) in any scenario. The evidence is in
`artifacts/screenshots/02_…` and `03_…`, which together show all 30 columns of
the Scenario A+B+C drift table: exactly four marked Detected, the rest Not
Detected with a drift score of 1.

Detection was this unambiguous because these are the **easiest possible drift
signal**: large, uniform, whole-column additive shifts applied to 100% of rows.
A K–S test compares empirical CDFs, and shifting an entire distribution 40,000
units along the axis separates the two CDFs almost completely. Realistic
production drift looks nothing like this — it is gradual, it affects a subset of
rows, it arrives as a slow change in the mix of counties submitting data rather
than a step change in a column. That kind of drift is genuinely hard to detect,
and a clean result here should not be read as evidence that drift detection is
easy in general.

### Output drift: detected twice out of three, and the miss is the finding

Prediction drift fired on Scenario A (p = 2.4e−5) and Scenario A+B+C
(p = 8e−29), both without any ground-truth labels — exactly the property that
makes it viable in production.

It **failed on Scenario A+B**, with a drift score of **0.999** — about as far
from significant as the test can report. This is not a defect in Evidently. It
is a correct report of a real fact: the two perturbations cancel (§7), so the
prediction distribution genuinely did not change. Uniform additive shifts in a
linear model leave the distribution's shape untouched and move only its
location, and here the two locations cancelled to within 0.53 deaths/100k.

The Test Suite makes the consequence stark. On Scenario A+B, **33 of 35 tests
pass** (`artifacts/screenshots/05_…`). The only two failures are the per-column
input drift tests for `medIncome` and `povertyPercent`. Every regression test —
RMSE, MAE, MAPE, Mean Error — passes. The prediction-drift test passes. A
production alerting rule built on "page me when accuracy degrades or predictions
shift" would have stayed **completely silent** while a third of the model's
inputs were physically impossible.

Contrast Scenario A+B+C, where 8 of 35 tests fail: four input-drift tests plus
Mean Error, MAE, RMSE and MAPE (`artifacts/screenshots/06_…`). That is what a
loud failure looks like, and it is the *easy* case.

### Which changes were detected clearly, which were not, and why

| Change | Input drift | Prediction drift | Accuracy tests | Bounds gate |
|---|---|---|---|---|
| A: `medIncome` −40,000 | **✓ clear** | ✓ | ✓ (Mean Error) | **✓ REJECT** — 191 rows negative |
| B: `povertyPercent` +20 | **✓ clear** | ✗ *(cancelled by A)* | ✗ *(cancelled by A)* | ✗ — 60.6% is a legal percentage |
| C: `AvgHouseholdSize` +2 | **✓ clear** | ✓ | ✓ (all four) | ✗ — 5.97 people is physically possible |

Every one of the three injected input changes was detected by input drift, in
every scenario in which it was present. What was *not* detected was the
**downstream effect** of B, and only because A was masking it. And the two checks
that would independently have caught A — bounds validation and prediction drift
— were both blind to B and C respectively.

**Why the undrifted target matters.** `TARGET_deathRate` shows p = 1.00
throughout, and that is the clue that identifies what kind of failure this is.
Genuine concept drift — the population actually changing — would move the inputs
*and* the outcome together. Inputs moving while the outcome sits perfectly still
is the signature of **data corruption**: a broken upstream ETL job, a units
change, a currency conversion applied twice. The distinction determines the fix.
Concept drift means retrain the model; data corruption means fix the pipeline,
and retraining on corrupted data would make everything worse. A monitoring setup
that watches inputs, predictions and (eventually) targets can tell these apart.
One that watches only accuracy cannot.

### Screenshots

Captured from the Evidently HTML reports, in `artifacts/screenshots/`:

| File | Shows |
|---|---|
| `01_scenario_abc_prediction_drift_detected.jpg` | A+B+C drift table head — `prediction` **Detected**, `TARGET_deathRate` **Not Detected** |
| `02_scenario_abc_drift_table_page1.jpg` | A+B+C, columns 1–20 — only `prediction` flagged |
| `03_scenario_abc_drift_table_page2_perturbed_columns.jpg` | A+B+C, columns 21–30 — `povertyPercent`, `medIncome`, `AvgHouseholdSize` all **Detected**, drift score 0 |
| `04_scenario_ab_prediction_drift_NOT_detected.jpg` | **The key exhibit.** A+B — `prediction` **Not Detected**, drift score **0.999106** |
| `05_scenario_ab_testsuite_2_of_35_failed.jpg` | A+B Test Suite — 33 pass, 2 fail; all regression tests green |
| `06_scenario_abc_testsuite_8_of_35_failed.jpg` | A+B+C Test Suite — 27 pass, 8 fail, including `prediction` drift |

---

## 9. Discussion questions

**How did each controlled input change affect model accuracy?**

Individually and cumulatively they behaved very differently. **A** (`medIncome`
−40,000) cost 0.041 R² (0.496 → 0.455) and moved every prediction down 5.69
deaths/100k. **B** (`povertyPercent` +20) added +5.16 per prediction, which
almost exactly *undid* A's effect and returned R² to 0.495 — a net accuracy
change from baseline of 0.001. **C** (`AvgHouseholdSize` +2) was catastrophic,
subtracting 14.92 per prediction and dropping R² from 0.495 to **0.193** — a 61%
loss against baseline, and since A and B had netted back to baseline, essentially
all of that loss is attributable to C alone.
The ordering is not intuitive from the size of the perturbations: a
40,000-unit shift did less damage than a 2-unit one, because the per-unit
coefficient on `AvgHouseholdSize` is 52,000× larger than the one on `medIncome`.
**Magnitude of input change is a poor predictor of impact; sensitivity is what
matters,** which argues for prioritising monitoring by feature importance rather
than by how much a column has moved.

**Which scenario caused the largest change in prediction behaviour?**

**A+B+C**, unambiguously, and by every measure: the largest mean-prediction shift
(−15.13), the worst accuracy (R² 0.193), the most drifted columns (4), and the
most failed tests (8 of 35).

But the more useful observation is that "most drift detected" and "most damage
done" **are not the same ranking**, and separating them is the operational
lesson. Scenario A+B has the same number of drifted columns as Scenario A (2) yet
does *less* accuracy damage than A. Scenario A+B has two input columns corrupted
— more than A — yet its predictions are indistinguishable from baseline. Drifted
column count measures how much the *input* changed; it says nothing about
whether the model cared. Ranking alerts by drift volume would have put A+B and A
at the same severity and both above their real impact, while a purely
impact-based ranking would have dismissed A+B entirely.

**Did Evidently detect the intended input changes? Why or why not?**

Yes — all three, in every scenario in which they were applied, with p-values at
the limits of floating-point representation (5e−278, 4e−257, 0.0) and no false
positives among the 25 untouched feature columns.

Detection was easy because the perturbations are the most favourable case a
drift detector can be handed: large uniform additive shifts applied to every row
of a column, compared against a clean reference of the same 610 counties. The
K–S statistic measures maximum separation between empirical CDFs, and shifting
an entire distribution bodily along the axis maximises exactly that quantity.
The scenarios also perturb columns by enormous margins relative to their natural
spread — 3.4, 3.1 and **7.9 training standard deviations** for `medIncome`,
`povertyPercent` and `AvgHouseholdSize` respectively — so the shifted
distributions barely overlap the reference at all. Real drift is gradual, partial and confounded
with legitimate population change, and would be far harder to distinguish. The
one thing the clean result *does* establish is that the reference set is sound:
the original-vs-original run produced 0 drifted columns and 0 failed tests, so
there is no baseline noise inflating these detections.

**Did the model output distribution change even when accuracy did not change
significantly?**

This assignment produced the sharpest possible version of this question, and the
answer is **no — and that is the problem.**

On **Scenario A+B**, accuracy did not change significantly (R² 0.495 vs 0.496,
RMSE 20.32 vs 20.31) and the output distribution *also* did not change
significantly (prediction drift p = 0.999, mean 178.25 vs 178.77, standard
deviation identical at 21.27). Both output-side signals were silent, in
agreement, and both were correct: the two perturbations cancelled, so the
predictions genuinely were normal. **Yet a third of the input rows contained
negative median incomes.** The model was producing sensible-looking numbers from
impossible data, and no amount of watching the output would have revealed it.

The converse case appears in **Scenario A**, where prediction drift was detected
(p = 2.4e−5) while the accuracy loss was mild (0.041 R²) — output drift fired
earlier and more sensitively than accuracy did. That is the usual argument for
prediction-drift monitoring and it holds here. A+B is the exception that shows
why it cannot be the *only* unsupervised check: when input errors offset, the
output looks clean, and only the input layer knows better.

**What monitoring checks would you keep if deploying this model in production?**

Four layers, ordered by how early each fires:

1. **Pre-inference schema and range validation** (implemented in
   `src/validation.py`). Reject rows with impossible values before scoring —
   this alone catches Scenario A's 191 negative incomes at the door, and it is
   the only check that acts *before* a bad prediction exists. Warn on values
   outside the training range rather than rejecting them, since extrapolation is
   a confidence problem rather than a validity one.
2. **Per-column input drift on the top-k features by importance.** Non-negotiable
   given the A+B result — it was the only check that fired. Weight the alerting
   by standardised coefficient rather than treating all 28 columns equally: a
   shift in `incidenceRate` (std coef 10.75) matters far more than the same
   shift in `studyPerCap`.
3. **Prediction drift**, because it needs no labels and fired earlier than
   accuracy on Scenario A. Kept as a complement to input drift, never as a
   substitute.
4. **Delayed regression performance**, recomputed whenever ground truth arrives.
   Retrospective by construction — useful for deciding whether to retrain,
   useless for catching a live incident.

I would also alert on **input drift and target drift jointly rather than
separately**, since it is the *combination* that diagnoses the problem: inputs
drifting with a stationary target means data corruption (fix the pipeline);
inputs and target drifting together means genuine population change (retrain).
The two demand opposite responses, and retraining on corrupted data actively
makes things worse.

**What additional safeguards would you add before using this model for
real-world decisions?**

- **Enforce the validation gate.** It runs report-only here to preserve the
  experiment; in production `enforce=True` should reject the 191 impossible rows
  outright and route them to a data-quality queue rather than scoring them.
- **Ship prediction intervals, not point estimates.** Every prediction should
  carry an uncertainty band, and any prediction derived from out-of-training-range
  inputs should be explicitly flagged as extrapolated. The §7 comparison is the
  argument: two defensible models disagree about the *sign* of the effect on
  out-of-range input, so a bare number conveys false precision.
- **A fallback path.** When inputs fail validation, return the county's
  historical death rate with a "model unavailable" flag rather than a prediction
  from corrupt data. A stale-but-honest answer beats a fresh wrong one.
- **Human review before any allocation decision.** The model's MAE is 8.4% of
  the mean and it systematically under-predicts the highest-mortality counties
  (§4) — precisely the counties most likely to need intervention. It should
  inform prioritisation, never determine it.
- **Fairness auditing across county composition.** `PctWhite`, `PctBlack`,
  `PctAsian`, `PctOtherRace` and the income columns are all features. Residuals
  must be checked for systematic bias across the racial and economic composition
  of counties before this model influences health-resource allocation, because a
  model that under-predicts mortality in predominantly Black or low-income
  counties would systematically divert resources away from them. Given the
  ethical weight of cancer-mortality predictions, this is a release blocker
  rather than a nice-to-have.
- **Address the ecological and spatial limitations.** Model spatial correlation
  explicitly (or at minimum split train/test by state rather than at random, so
  the test set actually measures generalisation to unseen regions), and document
  prominently that no individual-level inference is supported.
- **Scheduled retraining with drift-triggered review**, so the 2013 census
  baseline does not silently age into irrelevance.

---

## Operational lessons learned

1. **Output monitoring is not sufficient, and Scenario A+B proves it.** With 33
   of 35 tests passing, R² within 0.001 of baseline and prediction drift at
   p = 0.999, every output-side signal declared the model healthy while 31% of
   its input rows held negative incomes. Two independent input errors offset
   each other. Nothing downstream of the model could have known. **Monitor
   inputs, not just outputs** — this is the single most transferable finding here.

2. **Validation and drift detection catch different things, and neither is
   redundant.** The bounds gate caught Scenario A instantly and was blind to B
   and C, whose values are all individually legal. Drift detection caught all
   three but cannot distinguish "changed" from "impossible." The scenario that
   did the *most* damage (C) is the one a bounds check is least able to see.

3. **Sensitivity matters more than the size of the change.** A 40,000-unit shift
   in `medIncome` cost 0.041 R²; a 2-unit shift in `AvgHouseholdSize` cost 0.30.
   Alert thresholds and monitoring priorities should be weighted by the model's
   fitted sensitivity to each feature, not by how far the feature moved.

4. **Out-of-range model behaviour is arbitrary, so it must be prevented rather
   than characterised.** Linear regression and Random Forest, trained on
   identical data, moved their mean predictions in *opposite directions* on the
   same impossible inputs (163.6 vs 184.9). The forest fails quietly, the linear
   model fails loudly, and neither is right. There is no such thing as a model
   that degrades gracefully on impossible input — there is only a gate that
   stops it from seeing impossible input.

5. **A stationary target alongside drifting inputs is a diagnostic, not a
   footnote.** `TARGET_deathRate` held at p = 1.00 across all three scenarios,
   which is what distinguishes upstream data corruption from genuine population
   change. The two require opposite responses — fix the pipeline versus retrain
   the model — and retraining on corrupted inputs would compound the damage.
   Monitoring that tracks only model accuracy cannot make this distinction at all.

---

## Repository layout

```
assignment4/
├── README.md
├── requirements.txt
├── run_pipeline.py              # end-to-end driver (8 stages)
├── cancer_reg.csv               # raw dataset
├── data_dictionary.md
├── src/
│   ├── config.py                # paths, seed, excluded columns, bounds, scenarios
│   ├── data.py                  # Task 1 — load + validate
│   ├── preprocess.py            # Task 2 — clean + train/test split
│   ├── model.py                 # Task 3 — pipeline + Linear Regression training
│   ├── scenarios.py             # Task 6 — modified test sets + verification
│   ├── validation.py            # pre-inference bounds / OOD gate
│   ├── evaluate.py              # Tasks 4 & 7 — metrics + plots
│   ├── compare.py               # §3/§7 — linear vs random forest benchmark
│   └── monitoring.py            # Tasks 5 & 8 — Evidently reports
├── data/                        # train/test splits, scenario sets, scored frames
└── artifacts/
    ├── model.joblib
    ├── metrics/                 # JSON + CSV metric dumps
    ├── figures/                 # diagnostic, distribution + comparison plots
    ├── reports/                 # Evidently HTML (report_*.html, tests_*.html)
    └── screenshots/             # captured drift + test-suite evidence
```
