"""ADSP 31021 Assignment #3 -- Required H2O AutoML repeat workflow.

Runs H2O AutoML on the same cleaned athletes dataset / target / split used
for the PyCaret workflow, so results are directly comparable.

Usage:
    .venv-h2o/bin/python run_h2o_automl.py --features all
    .venv-h2o/bin/python run_h2o_automl.py --features top3

Saves to artifacts/: leaderboard CSV, variable-importance plot for the
leader model, held-out test-set metrics, and a JSON run summary.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config

RANDOM_SEED = config.RANDOM_SEED


def get_features(features_mode: str):
    if features_mode == "all":
        return config.ALL_FEATURES
    elif features_mode == "top3":
        return config.TOP3_FEATURES
    raise ValueError(features_mode)


def main(features_mode: str, max_runtime_secs: int, max_models: int, nfolds: int):
    import h2o
    from h2o.automl import H2OAutoML

    h2o.init(nthreads=-1, max_mem_size="4G")
    h2o.no_progress()

    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df = pd.read_csv(config.TEST_PATH)
    features = get_features(features_mode)
    print(f"[config] features_mode={features_mode}  features={features}")

    train_slice = train_df[features + [config.TARGET]].copy()
    test_slice = test_df[features + [config.TARGET]].copy()

    train_h2o = h2o.H2OFrame(train_slice)
    test_h2o = h2o.H2OFrame(test_slice)

    categorical_cols = [c for c in features if train_slice[c].dtype == object]
    for col in categorical_cols:
        train_h2o[col] = train_h2o[col].asfactor()
        test_h2o[col] = test_h2o[col].asfactor()

    aml = H2OAutoML(
        max_runtime_secs=max_runtime_secs,
        max_models=max_models,
        nfolds=nfolds,
        seed=RANDOM_SEED,
        sort_metric="RMSE",
        project_name=f"assignment3_h2o_{features_mode}_features",
    )

    train_start = time.time()
    aml.train(x=features, y=config.TARGET, training_frame=train_h2o)
    train_time = time.time() - train_start
    print(f"[automl train] completed in {train_time:.1f}s")

    from h2o.automl import get_leaderboard
    leaderboard = get_leaderboard(
        aml, extra_columns=["training_time_ms", "predict_time_per_row_ms"]
    ).as_data_frame()
    print(leaderboard.head(10).to_string())

    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    leaderboard_path = os.path.join(config.ARTIFACTS_DIR, f"h2o_leaderboard_{features_mode}.csv")
    leaderboard.to_csv(leaderboard_path, index=False)
    print(f"[save] leaderboard -> {leaderboard_path}")

    leader = aml.leader

    # Variable importance (tree-based / GLM leaders expose varimp)
    varimp_path = None
    try:
        varimp_df = leader.varimp(use_pandas=True)
        if varimp_df is not None:
            varimp_path = os.path.join(config.ARTIFACTS_DIR, f"h2o_varimp_{features_mode}.csv")
            varimp_df.to_csv(varimp_path, index=False)
            print(f"[save] variable importance -> {varimp_path}")
            print(varimp_df.head(5).to_string())
    except Exception as exc:
        print(f"[warn] leader model does not expose varimp (e.g. it's a Stacked Ensemble): {exc}")
        # Fall back to the best non-ensemble model in the leaderboard for importance
        for model_id in leaderboard["model_id"]:
            if "StackedEnsemble" not in model_id:
                fallback_model = h2o.get_model(model_id)
                try:
                    varimp_df = fallback_model.varimp(use_pandas=True)
                    varimp_path = os.path.join(config.ARTIFACTS_DIR, f"h2o_varimp_{features_mode}.csv")
                    varimp_df.to_csv(varimp_path, index=False)
                    print(f"[save] variable importance (fallback model {model_id}) -> {varimp_path}")
                    print(varimp_df.head(5).to_string())
                except Exception:
                    continue
                break

    # Held-out test-set evaluation
    predict_start = time.time()
    preds = leader.predict(test_h2o).as_data_frame()
    predict_time = time.time() - predict_start
    y_true = test_slice[config.TARGET].values
    y_pred = preds["predict"].values
    test_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    test_mae = float(mean_absolute_error(y_true, y_pred))
    test_r2 = float(r2_score(y_true, y_pred))
    print(f"[holdout test] RMSE={test_rmse:.2f}  MAE={test_mae:.2f}  R2={test_r2:.4f}  scoring_time={predict_time:.3f}s "
          f"({len(test_slice)} rows)")

    model_path = h2o.save_model(model=leader, path=config.ARTIFACTS_DIR, force=True)
    print(f"[save] leader model -> {model_path}")

    summary = {
        "platform": "H2O AutoML",
        "features_mode": features_mode,
        "features": features,
        "n_candidate_models": int(len(leaderboard)),
        "cv_folds": nfolds,
        "max_runtime_secs": max_runtime_secs,
        "max_models": max_models,
        "random_seed": RANDOM_SEED,
        "automl_train_time_sec": train_time,
        "leader_model_id": str(leader.model_id),
        "leader_cv_leaderboard_row": leaderboard.iloc[0].to_dict(),
        "holdout_test": {
            "rmse": test_rmse,
            "mae": test_mae,
            "r2": test_r2,
            "scoring_time_sec": predict_time,
            "n_test_rows": int(len(test_slice)),
        },
    }
    summary_path = os.path.join(config.ARTIFACTS_DIR, f"h2o_summary_{features_mode}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[save] run summary -> {summary_path}")

    h2o.cluster().shutdown(prompt=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", choices=["all", "top3"], default="all")
    parser.add_argument("--max_runtime_secs", type=int, default=300)
    parser.add_argument("--max_models", type=int, default=20)
    parser.add_argument("--nfolds", type=int, default=5)
    args = parser.parse_args()
    main(args.features, args.max_runtime_secs, args.max_models, args.nfolds)
