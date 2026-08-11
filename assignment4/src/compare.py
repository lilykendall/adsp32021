"""Model comparison — Linear Regression (the deployed model) vs Random Forest.

This module exists to support two claims in the write-up:

  1. What the linear assumption costs in baseline accuracy.
  2. How differently the two model families behave when the scenario shifts push
     inputs outside the training range. A tree ensemble cannot extrapolate — a
     `medIncome` of −15,965 lands in the same terminal leaf as the lowest income
     ever seen in training — whereas a linear model extrapolates without bound.
     That difference is invisible on the original test set and dominant on
     Scenario A+B+C.

Nothing here touches `artifacts/model.joblib`. The saved, scored and monitored
model remains the primary one from `src/model.py`.

Run standalone (assumes `python -m src.scenarios` has run):

    python -m src.compare
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score

if __package__ in (None, ""):  # allow `python src/compare.py` as well as `-m src.compare`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from . import config as cfg
from .evaluate import score
from .model import build_pipeline, train
from .preprocess import feature_columns


def load_scenario_frames() -> dict[str, pd.DataFrame]:
    frames = {}
    for name in cfg.SCENARIOS:
        path = cfg.DATA_DIR / (
            "test_original.csv" if name == "original" else f"test_{name}.csv"
        )
        frames[name] = pd.read_csv(path)
    return frames


def cross_validated_r2(train_df: pd.DataFrame, model: str) -> float:
    """5-fold CV R² on the training set — a fairer read on generalisation than
    the single test split, and it confirms the test-set gap is not a fluke."""
    features = feature_columns(train_df)
    scores = cross_val_score(
        build_pipeline(features, model=model),
        train_df[features],
        train_df[cfg.TARGET],
        cv=KFold(n_splits=5, shuffle=True, random_state=cfg.RANDOM_SEED),
        scoring="r2",
    )
    return float(scores.mean())


def plot_r2_paths(table: pd.DataFrame) -> None:
    """R² across the scenario sequence, one line per model — the single clearest
    picture of the extrapolation difference."""
    order = list(cfg.SCENARIOS)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for model in table["model"].unique():
        sub = table[table["model"] == model].set_index("dataset").loc[order]
        ax.plot(order, sub["r2"], marker="o", lw=2, label=model)
    ax.set_ylabel("R² on the test set")
    ax.set_xlabel("Scenario (perturbations applied cumulatively)")
    ax.set_title("Accuracy degradation path by model family")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([cfg.SCENARIO_LABELS[n].split(" (")[0] for n in order],
                       rotation=15, ha="right")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.FIGURE_DIR / "model_comparison_r2.png", dpi=140)
    plt.close(fig)


def main() -> None:
    train_df = pd.read_csv(cfg.DATA_DIR / "train.csv")
    frames = load_scenario_frames()

    rows = []
    for model in cfg.COMPARISON_MODELS:
        pipe = train(train_df, model=model)
        cv_r2 = cross_validated_r2(train_df, model)

        for name, df in frames.items():
            _, metrics = score(pipe, df)
            rows.append({
                "model": model,
                "dataset": name,
                "cv_r2_train": cv_r2 if name == "original" else np.nan,
                **metrics,
            })

    table = pd.DataFrame(rows)
    table.to_csv(cfg.METRIC_DIR / "model_comparison.csv", index=False)
    (cfg.METRIC_DIR / "model_comparison.json").write_text(
        json.dumps(rows, indent=2, default=float)
    )
    plot_r2_paths(table)

    baseline = table[table["dataset"] == "original"]
    print("Baseline accuracy (original test set):")
    print(baseline[["model", "rmse", "mae", "r2", "cv_r2_train"]]
          .round(4).to_string(index=False))

    print("\nR² across scenarios:")
    print(table.pivot(index="model", columns="dataset", values="r2")[list(cfg.SCENARIOS)]
          .round(4).to_string())

    print("\nMean prediction across scenarios:")
    print(table.pivot(index="model", columns="dataset", values="pred_mean")[list(cfg.SCENARIOS)]
          .round(2).to_string())

    print(f"\nWrote artifacts/metrics/model_comparison.csv|.json and "
          f"figures/model_comparison_r2.png")


if __name__ == "__main__":
    main()
