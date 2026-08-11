"""Task 2 — Preprocessing and train/test split.

Design rule for this assignment: the train/test split is done on the *raw*
feature frame, and every fitted preprocessing step (imputation, scaling) lives
inside the sklearn Pipeline built in `src/model.py`. That keeps two promises:

  1. No leakage — imputers/scalers are fit on training rows only.
  2. The scenario datasets in `src/scenarios.py` can be perturbed in the raw
     feature space (medIncome in dollars, povertyPercent in points), which is
     what the assignment spec describes and what Evidently should see.

Run standalone:

    python -m src.preprocess
"""

from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import train_test_split

if __package__ in (None, ""):  # allow `python src/preprocess.py` as well as `-m src.preprocess`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from . import config as cfg
from .data import load_raw


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Row/column-level cleaning applied to the full dataset before splitting.

    Only *stateless* operations belong here (dropping columns, nulling out
    impossible values). Anything that learns a statistic from the data must go
    in the model pipeline instead, or it leaks test information into training.
    """
    out = df.copy()

    # Drop identifier / redundant columns.
    out = out.drop(columns=[c for c in cfg.EXCLUDED_COLUMNS if c in out.columns])

    # Drop columns whose missingness makes them unusable.
    out = out.drop(columns=[c for c in cfg.HIGH_MISSING_DROP if c in out.columns])

    # Null out impossible values so the imputer handles them downstream.
    #
    # Nulling rather than dropping: the 30 MedianAge and 61 AvgHouseholdSize
    # offenders are recording-scale errors in *one* field of an otherwise sound
    # county record. Dropping ~91 rows would discard 28 valid features apiece,
    # and the errors are not missing-at-random — they cluster in small rural
    # counties, so dropping them would bias the training set. Nulling keeps every
    # county and confines the damage to the one untrustworthy cell.
    out.loc[out["MedianAge"] > cfg.MEDIAN_AGE_MAX_VALID, "MedianAge"] = pd.NA
    out.loc[
        out["AvgHouseholdSize"] < cfg.AVG_HOUSEHOLD_SIZE_MIN_VALID, "AvgHouseholdSize"
    ] = pd.NA

    return out


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80/20 train/test split with a fixed seed."""
    train, test = train_test_split(
        df, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_SEED, shuffle=True
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All modelling features = everything except the target."""
    return [c for c in df.columns if c != cfg.TARGET]


def main() -> None:
    raw = load_raw()
    cleaned = clean(raw)
    train, test = split(cleaned)

    train.to_csv(cfg.DATA_DIR / "train.csv", index=False)
    test.to_csv(cfg.DATA_DIR / "test_original.csv", index=False)

    manifest = {
        "raw_rows": int(len(raw)),
        "cleaned_rows": int(len(cleaned)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_size": cfg.TEST_SIZE,
        "random_seed": cfg.RANDOM_SEED,
        "target": cfg.TARGET,
        "excluded_columns": cfg.EXCLUDED_COLUMNS + cfg.HIGH_MISSING_DROP,
        "feature_columns": feature_columns(cleaned),
    }
    (cfg.METRIC_DIR / "split_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"raw {len(raw):,} -> cleaned {len(cleaned):,} rows, "
          f"{len(manifest['feature_columns'])} features")
    print(f"train {len(train):,} / test {len(test):,} "
          f"(test_size={cfg.TEST_SIZE}, seed={cfg.RANDOM_SEED})")
    print(f"Wrote data/train.csv, data/test_original.csv")


if __name__ == "__main__":
    main()
