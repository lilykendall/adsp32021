"""End-to-end training pipeline stages that read features from Feast.

The functions here are deliberately small and composable so they can be reused
by ``run_experiments.py`` (the 4-run experiment matrix) and tested in isolation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from feast import FeatureStore
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src import config


def get_store() -> FeatureStore:
    """Return a FeatureStore bound to the project's feature repo."""
    return FeatureStore(repo_path=config.FEATURE_REPO_DIR)


def _entity_df() -> pd.DataFrame:
    """Entity dataframe (keys + timestamps + label) driving the point-in-time join."""
    source = pd.read_parquet(config.FEATURES_PARQUET)
    return source[[config.ENTITY_KEY, "event_timestamp", config.TARGET]].copy()


def load_features(store: FeatureStore, feature_version: str) -> pd.DataFrame:
    """Retrieve a version's features from the offline store via its feature service.

    This is the feature-store *retrieval* step: the training code never touches
    the raw parquet columns directly, it asks Feast for the named service.
    """
    service_name = config.FEATURE_SERVICE_NAMES[feature_version]
    service = store.get_feature_service(service_name)
    training_df = store.get_historical_features(
        entity_df=_entity_df(), features=service
    ).to_df()
    feature_cols = config.FEATURE_VERSIONS[feature_version]
    training_df = training_df.dropna(subset=feature_cols + [config.TARGET])
    return training_df


def train_and_evaluate(training_df: pd.DataFrame, feature_version: str,
                       hyperparams: dict, seed: int = config.RANDOM_SEED):
    """Train a RandomForestRegressor and score it on a held-out split."""
    feature_cols = config.FEATURE_VERSIONS[feature_version]
    x = training_df[feature_cols]
    y = training_df[config.TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=config.TEST_SIZE, random_state=seed
    )
    model = RandomForestRegressor(random_state=seed, n_jobs=-1, **hyperparams)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
    }
    return model, metrics, (x_test, y_test, y_pred)
