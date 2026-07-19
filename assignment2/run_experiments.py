"""Run the 2x2 experiment matrix and track everything with MLflow.

Experiment design (per the assignment):
  * same algorithm everywhere  -> RandomForestRegressor
  * 2 feature versions         -> v1, v2  (served by Feast feature services)
  * 2 hyperparameter configs   -> hp_a, hp_b
  => 4 tracked runs.

Each run logs its feature version, hyperparameters, evaluation metrics and
diagnostic plots to MLflow. A comparison summary + chart are written to
artifacts/ at the end.
"""
from __future__ import annotations

import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd

from src import config, pipeline


def _pred_scatter_path(x_test, y_test, y_pred, tag: str) -> str:
    """Actual-vs-predicted scatter for a single run."""
    path = os.path.join(config.ARTIFACTS_DIR, f"pred_vs_actual_{tag}.png")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_test, y_pred, s=6, alpha=0.3)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_xlabel("Actual total_lift")
    ax.set_ylabel("Predicted total_lift")
    ax.set_title(f"Predicted vs actual ({tag})")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _importance_path(model, feature_cols, tag: str) -> str:
    """Feature-importance bar chart for a single run."""
    path = os.path.join(config.ARTIFACTS_DIR, f"feature_importance_{tag}.png")
    order = model.feature_importances_.argsort()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh([feature_cols[i] for i in order], model.feature_importances_[order])
    ax.set_title(f"Feature importance ({tag})")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def run_all() -> pd.DataFrame:
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    store = pipeline.get_store()

    # Retrieve each feature version once, reuse across hyperparameter configs.
    feature_frames = {
        v: pipeline.load_features(store, v) for v in config.FEATURE_VERSIONS
    }

    rows = []
    combos = itertools.product(config.FEATURE_VERSIONS, config.HYPERPARAMS)
    for feature_version, hp_name in combos:
        tag = f"{feature_version}_{hp_name}"
        hyperparams = config.HYPERPARAMS[hp_name]
        training_df = feature_frames[feature_version]

        with mlflow.start_run(run_name=tag):
            model, metrics, (x_test, y_test, y_pred) = pipeline.train_and_evaluate(
                training_df, feature_version, hyperparams
            )

            mlflow.set_tags({
                "feature_version": feature_version,
                "feature_service": config.FEATURE_SERVICE_NAMES[feature_version],
                "hp_config": hp_name,
                "algorithm": "RandomForestRegressor",
            })
            mlflow.log_params(hyperparams)
            mlflow.log_param("features", ",".join(config.FEATURE_VERSIONS[feature_version]))
            mlflow.log_param("n_features", len(config.FEATURE_VERSIONS[feature_version]))
            mlflow.log_metrics({k: v for k, v in metrics.items()})

            feature_cols = config.FEATURE_VERSIONS[feature_version]
            mlflow.log_artifact(_pred_scatter_path(x_test, y_test, y_pred, tag))
            mlflow.log_artifact(_importance_path(model, feature_cols, tag))
            mlflow.sklearn.log_model(
                model, artifact_path="model", input_example=x_test.head(3)
            )

            print(f"[{tag}] RMSE={metrics['rmse']:.2f} "
                  f"MAE={metrics['mae']:.2f} R2={metrics['r2']:.4f}")
            rows.append({
                "experiment": tag,
                "feature_version": feature_version,
                "hp_config": hp_name,
                **hyperparams,
                **metrics,
            })

    summary = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    summary.to_csv(os.path.join(config.ARTIFACTS_DIR, "experiment_summary.csv"), index=False)
    _summary_chart(summary)
    print("\n=== Experiment comparison (sorted by RMSE) ===")
    print(summary.to_string(index=False))
    return summary


def _summary_chart(summary: pd.DataFrame) -> None:
    """Grouped bar chart comparing RMSE and R2 across the four runs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.bar(summary["experiment"], summary["rmse"], color="#4C72B0")
    ax1.set_title("RMSE by experiment (lower is better)")
    ax1.tick_params(axis="x", rotation=30)
    ax2.bar(summary["experiment"], summary["r2"], color="#55A868")
    ax2.set_title("R2 by experiment (higher is better)")
    ax2.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(os.path.join(config.ARTIFACTS_DIR, "experiment_comparison.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    run_all()
