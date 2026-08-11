"""Tasks 4 & 7 — Baseline evaluation and scenario-based scoring.

Scores the one trained model against the original test set and each modified
test set, writes a tidy comparison table, and produces the diagnostic plots.

Run standalone:

    python -m src.evaluate
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if __package__ in (None, ""):  # allow `python src/evaluate.py` as well as `-m src.evaluate`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from . import config as cfg
from .model import load_model
from .preprocess import feature_columns


def score(model, df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Predict on `df` and return (predictions, metrics)."""
    features = feature_columns(df)
    y_true = df[cfg.TARGET]
    y_pred = model.predict(df[features])

    metrics = {
        "n": int(len(df)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        # Prediction-distribution stats — these are what change first when the
        # inputs shift, often before the accuracy metrics move much.
        "pred_mean": float(np.mean(y_pred)),
        "pred_std": float(np.std(y_pred)),
        "pred_min": float(np.min(y_pred)),
        "pred_max": float(np.max(y_pred)),
        "actual_mean": float(y_true.mean()),
    }
    return y_pred, metrics


def scored_frame(df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    """Test frame + a `prediction` column — the shape Evidently expects."""
    out = df.copy()
    out["prediction"] = y_pred
    return out


def plot_diagnostics(y_true, y_pred, name: str) -> None:
    """Predicted-vs-actual and residual plots for one dataset."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].scatter(y_true, y_pred, s=8, alpha=0.4)
    lo, hi = float(np.min(y_true)), float(np.max(y_true))
    axes[0].plot([lo, hi], [lo, hi], "r--", lw=1)
    axes[0].set_xlabel("Actual TARGET_deathRate")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title(f"Predicted vs actual — {name}")

    axes[1].scatter(y_pred, residuals, s=8, alpha=0.4)
    axes[1].axhline(0, color="r", ls="--", lw=1)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residual (actual − predicted)")
    axes[1].set_title(f"Residuals — {name}")

    fig.tight_layout()
    fig.savefig(cfg.FIGURE_DIR / f"diagnostics_{name}.png", dpi=140)
    plt.close(fig)


def plot_prediction_distributions(preds: dict[str, np.ndarray]) -> None:
    """Overlay prediction histograms across scenarios — the single clearest
    picture of output drift for the write-up."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, y_pred in preds.items():
        ax.hist(y_pred, bins=50, histtype="step", lw=1.6,
                label=cfg.SCENARIO_LABELS.get(name, name))
    ax.set_xlabel("Predicted TARGET_deathRate")
    ax.set_ylabel("Count")
    ax.set_title("Prediction distribution by scenario")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(cfg.FIGURE_DIR / "prediction_distributions.png", dpi=140)
    plt.close(fig)


def main() -> None:
    model = load_model()
    results, predictions = {}, {}

    for name in cfg.SCENARIOS:
        path = cfg.DATA_DIR / (
            "test_original.csv" if name == "original" else f"test_{name}.csv"
        )
        df = pd.read_csv(path)
        y_pred, metrics = score(model, df)

        results[name] = metrics
        predictions[name] = y_pred
        scored_frame(df, y_pred).to_csv(cfg.DATA_DIR / f"scored_{name}.csv", index=False)
        plot_diagnostics(df[cfg.TARGET], y_pred, name)

    plot_prediction_distributions(predictions)

    table = pd.DataFrame(results).T
    table.index.name = "dataset"
    table.to_csv(cfg.METRIC_DIR / "scenario_metrics.csv")
    (cfg.METRIC_DIR / "scenario_metrics.json").write_text(json.dumps(results, indent=2))

    print(table[["n", "rmse", "mae", "r2", "pred_mean", "pred_std"]].round(3).to_string())
    print(f"\nWrote artifacts/metrics/scenario_metrics.csv and figures/")


if __name__ == "__main__":
    main()
