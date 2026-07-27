"""ADSP 31021 Assignment #3 -- Chosen MLOps platform AutoML workflow.

Platform: PyCaret (low-code AutoML) with MLflow as the experiment-tracking
backend (`log_experiment='mlflow'`), continuing the MLflow setup used in
Assignment #2.

Usage:
    .venv-pycaret/bin/python run_pycaret_automl.py --features all
    .venv-pycaret/bin/python run_pycaret_automl.py --features top3

Each run performs, and saves to artifacts/:
  - the AutoML leaderboard (all candidate models, CV validation scores + fit time)
  - feature importance plot for the best model
  - held-out test-set metrics (RMSE/MAE/R2) for the top model, comparable to
    Assignment #1's baseline evaluation
  - a JSON run summary (config, timings, best model)

All runs are logged as MLflow experiments under ./mlruns (view with
`mlflow ui --backend-store-uri ./mlruns`).
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlflow.db")
os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{_DB_PATH}")

from src import config  # noqa: E402

RANDOM_SEED = config.RANDOM_SEED
CATEGORICAL_ALL = ["region", "gender", "howlong", "background", "experience", "schedule", "eat"]
NUMERIC_ALL = ["age", "height", "weight"]


def get_feature_config(features_mode: str):
    if features_mode == "all":
        features = config.ALL_FEATURES
        categorical = [c for c in CATEGORICAL_ALL if c in features]
        numeric = [c for c in NUMERIC_ALL if c in features]
    elif features_mode == "top3":
        features = config.TOP3_FEATURES
        categorical = [c for c in CATEGORICAL_ALL if c in features]
        numeric = [c for c in NUMERIC_ALL if c in features]
    else:
        raise ValueError(features_mode)
    return features, categorical, numeric


def main(features_mode: str, fold: int, n_select: int):
    from pycaret.regression import (
        setup, compare_models, pull, plot_model, predict_model, save_model, finalize_model,
    )

    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df = pd.read_csv(config.TEST_PATH)

    features, categorical, numeric = get_feature_config(features_mode)
    print(f"[config] features_mode={features_mode}  features={features}")

    train_slice = train_df[features + [config.TARGET]].copy()
    test_slice = test_df[features + [config.TARGET]].copy()

    experiment_name = f"assignment3_pycaret_{features_mode}_features"
    run_tag = f"pycaret_{features_mode}"

    setup_start = time.time()
    reg_setup = setup(
        data=train_slice,
        target=config.TARGET,
        session_id=RANDOM_SEED,
        train_size=0.8,
        fold=fold,
        categorical_features=categorical,
        numeric_features=numeric,
        log_experiment="mlflow",
        experiment_name=experiment_name,
        log_plots=True,
        verbose=False,
    )
    setup_time = time.time() - setup_start
    print(f"[setup] completed in {setup_time:.1f}s")

    compare_start = time.time()
    top_models = compare_models(sort="RMSE", turbo=True, n_select=n_select, verbose=False)
    compare_time = time.time() - compare_start
    leaderboard = pull()
    print(f"[compare_models] completed in {compare_time:.1f}s over {len(leaderboard)} candidate models")
    print(leaderboard[["Model", "RMSE", "MAE", "R2", "TT (Sec)"]].head(10).to_string())

    best_model = top_models[0] if isinstance(top_models, list) else top_models

    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    leaderboard_path = os.path.join(config.ARTIFACTS_DIR, f"pycaret_leaderboard_{features_mode}.csv")
    leaderboard.to_csv(leaderboard_path, index=False)
    print(f"[save] leaderboard -> {leaderboard_path}")

    # Feature importance for the best model (tree-based models expose feature_importances_)
    fi_path = None
    try:
        plot_model(best_model, plot="feature", save=config.ARTIFACTS_DIR)
        default_name = "Feature Importance.png"
        src_path = os.path.join(config.ARTIFACTS_DIR, default_name)
        if os.path.exists(src_path):
            fi_path = os.path.join(config.ARTIFACTS_DIR, f"pycaret_feature_importance_{features_mode}.png")
            os.replace(src_path, fi_path)
            print(f"[save] feature importance plot -> {fi_path}")
    except Exception as exc:  # best model may not support native feature importance
        print(f"[warn] could not plot feature importance for best model: {exc}")

    # Held-out test-set evaluation (comparable to Assignment #1's RMSE/MAE/R2)
    predict_start = time.time()
    holdout_preds = predict_model(best_model, data=test_slice)
    predict_time = time.time() - predict_start
    pred_col = "prediction_label" if "prediction_label" in holdout_preds.columns else "Label"
    y_true = holdout_preds[config.TARGET]
    y_pred = holdout_preds[pred_col]
    test_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    test_mae = float(mean_absolute_error(y_true, y_pred))
    test_r2 = float(r2_score(y_true, y_pred))
    print(f"[holdout test] RMSE={test_rmse:.2f}  MAE={test_mae:.2f}  R2={test_r2:.4f}  scoring_time={predict_time:.3f}s "
          f"({len(test_slice)} rows)")

    model_path = os.path.join(config.ARTIFACTS_DIR, f"pycaret_best_model_{features_mode}")
    save_model(best_model, model_path, verbose=False)

    summary = {
        "platform": "PyCaret (MLflow backend)",
        "features_mode": features_mode,
        "features": features,
        "n_candidate_models": int(len(leaderboard)),
        "cv_folds": fold,
        "random_seed": RANDOM_SEED,
        "setup_time_sec": setup_time,
        "compare_models_time_sec": compare_time,
        "best_model": str(best_model.__class__.__name__),
        "best_model_cv_leaderboard_row": leaderboard.iloc[0].to_dict(),
        "holdout_test": {
            "rmse": test_rmse,
            "mae": test_mae,
            "r2": test_r2,
            "scoring_time_sec": predict_time,
            "n_test_rows": int(len(test_slice)),
        },
        "mlflow_experiment_name": experiment_name,
    }
    summary_path = os.path.join(config.ARTIFACTS_DIR, f"pycaret_summary_{features_mode}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[save] run summary -> {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", choices=["all", "top3"], default="all")
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--n_select", type=int, default=3)
    args = parser.parse_args()
    main(args.features, args.fold, args.n_select)
