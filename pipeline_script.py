
"""ADSP 31021 Assignment 1 — Main pipeline script."""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
TEST_SIZE = 0.2
RAW_DATA_PATH = "athletes.csv"
DATA_DIR = "data"
V1_PATH = os.path.join(DATA_DIR, "athletes_v1.csv")
V2_PATH = os.path.join(DATA_DIR, "athletes_v2.csv")
TARGET = "total_lift"
V2_FEATURES = ["age", "weight", "height", "gender_enc", "howlong_enc", "schedule_enc"]

np.random.seed(RANDOM_SEED)


def stage_ingest(raw_path):
    """Load raw CSV."""
    return pd.read_csv(raw_path)


def stage_clean(df):
    """Apply prescribed data cleaning steps."""
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
            "affiliate", "team", "name", "athlete_id",
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
    data = data.dropna(
        subset=["background", "experience", "schedule", "howlong", "eat"]
    )
    return data


def stage_feature_engineer(df):
    """Compute total_lift and encode categorical columns."""
    data = df.copy()
    data["total_lift"] = (
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
    return data


def stage_train(df, features, target, seed=RANDOM_SEED, test_size=TEST_SIZE):
    """Split data and train a RandomForestRegressor."""
    x_data = df[features]
    y_data = df[target]
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=test_size, random_state=seed
    )
    model = RandomForestRegressor(
        n_estimators=200, max_depth=15, random_state=seed, n_jobs=-1
    )
    model.fit(x_train, y_train)
    return model, x_test, y_test


def stage_evaluate(model, x_test, y_test):
    """Compute RMSE, MAE, R² on the held-out test set."""
    y_pred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def run_pipeline(raw_path, features, target):
    """Execute all five pipeline stages end-to-end."""
    os.makedirs(DATA_DIR, exist_ok=True)
    raw_df = stage_ingest(raw_path)
    clean_df = stage_clean(raw_df)
    feat_df = stage_feature_engineer(clean_df)
    model, x_test, y_test = stage_train(feat_df, features, target)
    metrics = stage_evaluate(model, x_test, y_test)
    print(f"RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}  R2={metrics['r2']:.4f}")
    return model, metrics


if __name__ == "__main__":
    run_pipeline(RAW_DATA_PATH, V2_FEATURES, TARGET)
