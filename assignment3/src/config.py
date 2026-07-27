"""Single source of truth: paths, seeds, target/feature definitions.

Cleaning rules, target definition, and split strategy are carried over
unchanged from Assignment #1 (`pipeline_script.py`) / Assignment #2
(`src/preprocess.py`) so AutoML results here are comparable to the earlier
baseline model.
"""
import os

RANDOM_SEED = 42
TEST_SIZE = 0.20

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "athletes.csv")
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
PROCESSED_PATH = os.path.join(DATA_DIR, "athletes_processed.csv")
TRAIN_PATH = os.path.join(DATA_DIR, "athletes_train.csv")
TEST_PATH = os.path.join(DATA_DIR, "athletes_test.csv")

TARGET = "total_lift"

# "All appropriate features" for AutoML: every raw, non-leaking predictor
# that survives Assignment #1's cleaning step. Unlike Assignment #1/#2 (which
# hand-engineered `gender_enc`/`howlong_enc`/`schedule_enc` and used only a
# 4-6 column subset), we hand AutoML the *raw* categorical columns too and
# let it perform its own encoding -- that is precisely the part of the
# workflow AutoML is meant to automate.
ALL_FEATURES = [
    "region", "gender", "age", "height", "weight",
    "howlong", "background", "experience", "schedule", "eat",
]

# Columns that make up the target itself (would leak) or are benchmark WOD
# times/identity columns dropped during cleaning -- excluded from features.
LEAKAGE_COLUMNS = ["deadlift", "candj", "snatch", "backsq"]
DROPPED_COLUMNS = [
    "affiliate", "team", "name", "athlete_id",
    "fran", "helen", "grace", "filthy50",
    "fgonebad", "run400", "run5k", "pullups", "train",
]

# Top-3 feature set, fixed after inspecting the "all features" AutoML run's
# reported feature importance: the PyCaret LightGBM leader (see
# artifacts/pycaret_feature_importance_all.png) ranked weight > age > height
# > background > schedule. Reused by every reduced-feature run so the
# comparison is apples-to-apples.
TOP3_FEATURES = ["weight", "age", "height"]
