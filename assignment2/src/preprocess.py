"""Data ingestion, cleaning and feature engineering.

This stage turns the raw ``athletes.csv`` into a single tidy parquet file that
backs the Feast offline store. Every candidate feature for *all* feature
versions is materialised here; the feature store is then responsible for
serving the version-specific subset to the training code.

Cleaning assumptions (documented in the README) are inherited from Assignment 1
so results remain comparable across the two assignments.
"""
from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd

from src import config

warnings.filterwarnings("ignore")

# A fixed event timestamp. The dataset is a static snapshot with no natural
# event time, so we assign one constant timestamp to every athlete. This lets
# Feast perform its (trivial) point-in-time join deterministically.
EVENT_TIMESTAMP = pd.Timestamp("2024-01-01 00:00:00")

_HOWLONG_MAP = {
    "Less than 6 months|": 1,
    "6-12 months|": 2,
    "1-2 years|": 3,
    "2-4 years|": 4,
    "4+ years|": 5,
}
_SCHEDULE_MAP = {
    "1 day per week|": 1,
    "2 days per week|": 2,
    "3 days per week|": 3,
    "4 days per week|": 4,
    "5+ days per week|": 5,
    "I do multiple workouts in a day 2x a week|": 5,
    "I do multiple workouts in a day 3x a week|": 6,
    "I do multiple workouts in a day 4x a week|": 7,
}


def ingest(raw_path: str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV (data ingestion stage)."""
    return pd.read_csv(raw_path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unusable rows/columns and clip physiologically implausible values."""
    data = df.copy()
    data = data.dropna(
        subset=[
            "region", "age", "weight", "height", "howlong",
            "gender", "eat", "background", "experience",
            "schedule", "deadlift", "candj", "snatch", "backsq",
        ]
    )
    data = data.drop(
        columns=[
            "affiliate", "team", "name",
            "fran", "helen", "grace", "filthy50",
            "fgonebad", "run400", "run5k", "pullups", "train",
        ],
        errors="ignore",
    )
    data = data[data["weight"] < 1500]
    data = data[data["gender"] != "--"]
    data = data[data["age"] >= 18]
    data = data[(data["height"] < 96) & (data["height"] > 48)]
    data = data[
        ((data["gender"] == "Male") & (data["deadlift"] <= 1105))
        | ((data["gender"] == "Female") & (data["deadlift"] <= 636))
    ]
    data = data[(data["candj"] > 0) & (data["candj"] <= 395)]
    data = data[(data["snatch"] > 0) & (data["snatch"] <= 496)]
    data = data[(data["backsq"] > 0) & (data["backsq"] <= 1069)]
    data = data.replace({"Decline to answer|": np.nan})
    data = data.dropna(subset=["background", "experience", "schedule", "howlong", "eat"])
    return data


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the target and encode categorical features for both versions."""
    data = df.copy()
    data["total_lift"] = (
        data["deadlift"] + data["candj"] + data["snatch"] + data["backsq"]
    )
    data["gender_enc"] = (data["gender"] == "Male").astype(int)
    data["howlong_enc"] = data["howlong"].map(_HOWLONG_MAP).fillna(3).astype(int)
    data["schedule_enc"] = data["schedule"].map(_SCHEDULE_MAP).fillna(3).astype(int)

    # Feast requires a unique entity key and an event timestamp column.
    data[config.ENTITY_KEY] = data[config.ENTITY_KEY].astype("int64")
    data = data.drop_duplicates(subset=[config.ENTITY_KEY])
    data["event_timestamp"] = EVENT_TIMESTAMP

    keep = (
        [config.ENTITY_KEY, "event_timestamp", config.TARGET]
        + config.FEATURE_VERSIONS["v2"]
    )
    return data[keep].reset_index(drop=True)


def build_feature_parquet(raw_path: str = config.RAW_DATA_PATH,
                          out_path: str = config.FEATURES_PARQUET) -> pd.DataFrame:
    """Run ingest -> clean -> engineer and persist the Feast source parquet."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    frame = engineer(clean(ingest(raw_path)))
    frame.to_parquet(out_path, index=False)
    print(f"[preprocess] wrote {len(frame):,} rows x {frame.shape[1]} cols -> {out_path}")
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the Feast feature parquet.")
    parser.add_argument("--raw", default=config.RAW_DATA_PATH)
    parser.add_argument("--out", default=config.FEATURES_PARQUET)
    args = parser.parse_args()
    build_feature_parquet(args.raw, args.out)
