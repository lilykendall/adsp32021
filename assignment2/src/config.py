"""Central configuration for the ADSP 31021 Assignment 2 pipeline.

Keeping all paths, seeds and feature-version definitions in one module keeps the
pipeline reproducible and lets every stage import a single source of truth.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.2

# ---------------------------------------------------------------------------
# Paths (all relative to the assignment2/ project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "athletes.csv")

FEATURE_REPO_DIR = os.path.join(PROJECT_ROOT, "feature_repo")
FEATURE_DATA_DIR = os.path.join(FEATURE_REPO_DIR, "data")
# Parquet that backs the Feast offline (file) store.
FEATURES_PARQUET = os.path.join(FEATURE_DATA_DIR, "athlete_features.parquet")

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
MLFLOW_TRACKING_URI = "file://" + os.path.join(PROJECT_ROOT, "mlruns")
MLFLOW_EXPERIMENT = "crossfit_total_lift"

# ---------------------------------------------------------------------------
# Modelling target
# ---------------------------------------------------------------------------
TARGET = "total_lift"
ENTITY_KEY = "athlete_id"

# ---------------------------------------------------------------------------
# Feature versions
# ---------------------------------------------------------------------------
# Two versions of the feature *definition*. They are registered in Feast as two
# separate feature services (see feature_repo/definitions.py). The lists below
# mirror those services and are used by the training code to slice the frame
# returned from the feature store.
#
#   v1 - "physical baseline": demographic / anthropometric signals only.
#   v2 - "physical + training behaviour": v1 plus engineered training-engagement
#        features (how long the athlete has trained, weekly schedule intensity).
FEATURE_VERSIONS = {
    "v1": ["age", "weight", "height", "gender_enc"],
    "v2": ["age", "weight", "height", "gender_enc", "howlong_enc", "schedule_enc"],
}

FEATURE_SERVICE_NAMES = {
    "v1": "athlete_service_v1",
    "v2": "athlete_service_v2",
}

# ---------------------------------------------------------------------------
# Hyperparameter configurations (same algorithm: RandomForestRegressor)
# ---------------------------------------------------------------------------
HYPERPARAMS = {
    "hp_a": {"n_estimators": 100, "max_depth": 8},
    "hp_b": {"n_estimators": 300, "max_depth": 20},
}
