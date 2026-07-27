"""Ingest + clean the athletes dataset and produce train/test splits.

Cleaning logic is copied verbatim from Assignment #1's `pipeline_script.py`
so the target definition (`total_lift`) and row filters are identical,
keeping this assignment's AutoML results comparable to the Assignment #1
baseline model.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def stage_ingest(raw_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    print(f"[ingest] loaded {df.shape[0]:,} rows from {raw_path}")
    return df


def stage_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Identical row filters to Assignment #1 / #2."""
    data = df.copy()
    data = data.dropna(
        subset=[
            "region", "age", "weight", "height", "howlong",
            "gender", "eat", "background", "experience",
            "schedule", "deadlift", "candj", "snatch", "backsq",
        ]
    )
    data = data.drop(columns=config.DROPPED_COLUMNS, errors="ignore")
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
    print(f"[clean] {data.shape[0]:,} rows remain after cleaning")
    return data


def stage_feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the target. Raw categoricals are kept as-is (AutoML encodes
    them itself); the Assignment #1-style encoded columns are also added so
    they remain available for direct comparison if needed."""
    data = df.copy()
    data[config.TARGET] = (
        data["deadlift"] + data["candj"] + data["snatch"] + data["backsq"]
    )

    data["gender_enc"] = (data["gender"] == "Male").astype(int)
    howlong_map = {
        "Less than 6 months|": 1, "6-12 months|": 2, "1-2 years|": 3,
        "2-4 years|": 4, "4+ years|": 5,
    }
    schedule_map = {
        "1 day per week|": 1, "2 days per week|": 2, "3 days per week|": 3,
        "4 days per week|": 4, "5+ days per week|": 5,
        "I do multiple workouts in a day 2x a week|": 5,
        "I do multiple workouts in a day 3x a week|": 6,
        "I do multiple workouts in a day 4x a week|": 7,
    }
    data["howlong_enc"] = data["howlong"].map(howlong_map).fillna(3).astype(int)
    data["schedule_enc"] = data["schedule"].map(schedule_map).fillna(3).astype(int)
    print(f"[feature_engineer] {config.TARGET} range: [{data[config.TARGET].min()}, {data[config.TARGET].max()}]")
    return data


def build_processed_dataset() -> pd.DataFrame:
    raw_df = stage_ingest(config.RAW_DATA_PATH)
    clean_df = stage_clean(raw_df)
    feat_df = stage_feature_engineer(clean_df)
    return feat_df


def build_and_save_splits():
    """Build the processed dataset and write processed/train/test CSVs,
    using the same 80/20, seed=42 split as Assignment #1."""
    import os
    os.makedirs(config.DATA_DIR, exist_ok=True)

    feat_df = build_processed_dataset()
    feat_df.to_csv(config.PROCESSED_PATH, index=False)

    train_df, test_df = train_test_split(
        feat_df, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
    )
    train_df.to_csv(config.TRAIN_PATH, index=False)
    test_df.to_csv(config.TEST_PATH, index=False)
    print(f"[split] train={train_df.shape[0]:,}  test={test_df.shape[0]:,}")
    return train_df, test_df


if __name__ == "__main__":
    build_and_save_splits()
