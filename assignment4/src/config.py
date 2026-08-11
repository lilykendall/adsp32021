"""Central configuration for Assignment #4 — paths, seeds, columns, scenarios.

Everything that another module needs to *agree on* lives here, so the pipeline
stays reproducible and there is exactly one place to change a decision.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent

RAW_CSV = ROOT / "cancer_reg.csv"
RAW_ENCODING = "latin-1"  # file is not UTF-8 (county names contain 0xF1 = "ñ")

DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ARTIFACT_DIR / "reports"       # Evidently HTML reports
FIGURE_DIR = ARTIFACT_DIR / "figures"       # matplotlib PNGs
METRIC_DIR = ARTIFACT_DIR / "metrics"       # JSON / CSV metric dumps
MODEL_PATH = ARTIFACT_DIR / "model.joblib"

for _d in (DATA_DIR, ARTIFACT_DIR, REPORT_DIR, FIGURE_DIR, METRIC_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42
TEST_SIZE = 0.20

# --------------------------------------------------------------------------- #
# Target
# --------------------------------------------------------------------------- #
TARGET = "TARGET_deathRate"

# --------------------------------------------------------------------------- #
# Feature handling
# --------------------------------------------------------------------------- #
# Columns excluded from modelling. Each exclusion is justified in README §1.
EXCLUDED_COLUMNS = [
    "Geography",   # free-text county identifier — 3,047 unique values, no signal
    "binnedInc",   # decile *string* of medIncome; redundant with medIncome and
                   # would mask the Scenario A income shift if kept
    "avgDeathsPerYear",  # target leakage: it is the numerator of the target, and
                         # popEst2015 (the denominator) is in the feature set, so
                         # the model could nearly reconstruct TARGET_deathRate
    # "avgAnnCount",     # KEPT. It is the count of *diagnoses*, not deaths — a
                         # genuine predictor available before any death occurs.
                         # See README §1 for the full leakage argument.
]

# Columns with heavy missingness. Decision: drop the two worst, impute the rest.
# PctSomeCol18_24         -> 2,285 / 3,047 missing (75%) — dropped; imputation
#                            would fabricate three quarters of the column
# PctPrivateCoverageAlone ->   609 / 3,047 missing (20%) — dropped; also almost
#                            perfectly collinear with PctPrivateCoverage
# PctEmployed16_Over      ->   152 / 3,047 missing (5%) — KEPT and median-imputed;
#                            low missingness and a useful predictor
HIGH_MISSING_DROP = [
    "PctSomeCol18_24",
    "PctPrivateCoverageAlone"
]

# Known invalid values found during validation (see src/data.py::validate).
#   MedianAge         -> max 624 (impossible; a handful of rows are ~10x scaled)
#   AvgHouseholdSize  -> min 0.0221 (recorded as a fraction, not a count)
#
# Only 30 and 61 rows respectively fall outside the valid range, so they are
# nulled out in preprocess.clean() and median-imputed downstream rather than
# dropped — that keeps every county in the dataset. See README §2.
MEDIAN_AGE_MAX_VALID = 100.0
AVG_HOUSEHOLD_SIZE_MIN_VALID = 1.0

# --------------------------------------------------------------------------- #
# Model selection
# --------------------------------------------------------------------------- #
# `PRIMARY_MODEL` is the model that is trained, saved to artifacts/model.joblib,
# and monitored throughout the assignment. `COMPARISON_MODELS` are additionally
# trained by `src/compare.py` purely to produce the benchmark table in README §3
# — they are never monitored and never overwrite the saved model.
PRIMARY_MODEL = "linear"
COMPARISON_MODELS = ["linear", "random_forest"]

# --------------------------------------------------------------------------- #
# Pre-inference input validation (see src/validation.py)
# --------------------------------------------------------------------------- #
# Hard physical bounds: a value outside these is *impossible*, not merely
# unusual, and a production gate should refuse to score the row. `None` means
# unbounded on that side. Any column not listed here and matching the Pct/Percent
# naming convention gets an automatic [0, 100] bound (see validation.py).
INPUT_BOUNDS = {
    "medIncome": (0.0, None),            # income cannot be negative
    "povertyPercent": (0.0, 100.0),      # a percentage
    "AvgHouseholdSize": (AVG_HOUSEHOLD_SIZE_MIN_VALID, 10.0),
    "MedianAge": (0.0, MEDIAN_AGE_MAX_VALID),
    "MedianAgeMale": (0.0, 100.0),
    "MedianAgeFemale": (0.0, 100.0),
    "popEst2015": (0.0, None),
    "incidenceRate": (0.0, None),
    "avgAnnCount": (0.0, None),
    "studyPerCap": (0.0, None),
    "BirthRate": (0.0, None),
}

# Fraction of a column's values that may fall outside its *training* range
# before the gate flags it as out-of-distribution. Distinct from a hard-bound
# violation: out-of-distribution values are possible, just never seen in training.
OOD_ALERT_THRESHOLD = 0.05

# --------------------------------------------------------------------------- #
# Monitoring scenarios (assignment spec, applied cumulatively)
# --------------------------------------------------------------------------- #
# NOTE: the spec writes the column names lowercased ("medianincome"). The actual
# CSV headers are `medIncome`, `povertyPercent`, `AvgHouseholdSize`.
SCENARIO_A = {"medIncome": -40_000}
SCENARIO_B = {"povertyPercent": +20.0}
SCENARIO_C = {"AvgHouseholdSize": +2.0}

SCENARIOS = {
    "original": {},
    "scenario_a": {**SCENARIO_A},
    "scenario_ab": {**SCENARIO_A, **SCENARIO_B},
    "scenario_abc": {**SCENARIO_A, **SCENARIO_B, **SCENARIO_C},
}

SCENARIO_LABELS = {
    "original": "Original test set",
    "scenario_a": "Scenario A (medIncome −40,000)",
    "scenario_ab": "Scenario A+B (… povertyPercent +20)",
    "scenario_abc": "Scenario A+B+C (… AvgHouseholdSize +2)",
}
