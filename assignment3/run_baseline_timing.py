"""Re-time the Assignment #1 Part A baseline model on this assignment's
identical train/test split, since the original notebook reported RMSE/MAE/R2
but never logged fit/predict wall-clock time. Model, hyperparameters, and
features (age, weight, height, gender_enc) are unchanged from Assignment #1.
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config

BASELINE_FEATURES = ["age", "weight", "height", "gender_enc"]

train_df = pd.read_csv(config.TRAIN_PATH)
test_df = pd.read_csv(config.TEST_PATH)

x_train, y_train = train_df[BASELINE_FEATURES], train_df[config.TARGET]
x_test, y_test = test_df[BASELINE_FEATURES], test_df[config.TARGET]

model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=config.RANDOM_SEED, n_jobs=-1)

fit_start = time.time()
model.fit(x_train, y_train)
fit_time = time.time() - fit_start

predict_start = time.time()
y_pred = model.predict(x_test)
predict_time = time.time() - predict_start

rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae = float(mean_absolute_error(y_test, y_pred))
r2 = float(r2_score(y_test, y_pred))

print(f"[baseline retimed] RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.4f}  "
      f"fit_time={fit_time:.3f}s  predict_time={predict_time:.3f}s  "
      f"(train={len(x_train):,} test={len(x_test):,})")

summary = {
    "model": "RandomForestRegressor (Assignment #1 Part A baseline, retimed)",
    "features": BASELINE_FEATURES,
    "rmse": rmse,
    "mae": mae,
    "r2": r2,
    "fit_time_sec": fit_time,
    "predict_time_sec": predict_time,
    "n_train_rows": len(x_train),
    "n_test_rows": len(x_test),
}
os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
with open(os.path.join(config.ARTIFACTS_DIR, "baseline_retimed_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
