"""Task 1 — Dataset loading and data validation.

Run standalone to print the validation summary that feeds README §1:

    python -m src.data
"""

from __future__ import annotations

import json

import pandas as pd

if __package__ in (None, ""):  # allow `python src/data.py` as well as `-m src.data`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from . import config as cfg


def load_raw() -> pd.DataFrame:
    """Load the raw cancer dataset exactly as shipped (no cleaning)."""
    return pd.read_csv(cfg.RAW_CSV, encoding=cfg.RAW_ENCODING)


def validate(df: pd.DataFrame) -> dict:
    """Collect the basic data-validation facts the rubric asks for.

    Returns a JSON-serialisable dict: shape, dtypes, missingness, duplicates,
    and a hand-picked set of range checks for obviously invalid values.
    """
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    numeric = df.select_dtypes("number")

    summary = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "target": cfg.TARGET,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "categorical_columns": list(df.select_dtypes("object").columns),
        "missing_counts": {k: int(v) for k, v in missing.items()},
        "missing_pct": {k: round(100 * v / len(df), 2) for k, v in missing.items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "constant_columns": [c for c in df.columns if df[c].nunique(dropna=False) <= 1],
        # Obvious invalid values.
        "invalid_values": {
            "MedianAge_gt_100": int((df["MedianAge"] > cfg.MEDIAN_AGE_MAX_VALID).sum()),
            "AvgHouseholdSize_lt_1": int(
                (df["AvgHouseholdSize"] < cfg.AVG_HOUSEHOLD_SIZE_MIN_VALID).sum()
            ),
            "negative_values": {
                c: int((numeric[c] < 0).sum())
                for c in numeric.columns
                if (numeric[c] < 0).any()
            },
            # Percentage columns must lie in [0, 100] by definition.
            "percent_columns_out_of_range": {
                c: int(((numeric[c] < 0) | (numeric[c] > 100)).sum())
                for c in numeric.columns
                if (c.startswith("Pct") or c.startswith("Percent"))
                and ((numeric[c] < 0) | (numeric[c] > 100)).any()
            },
            # studyPerCap is zero for most counties (no clinical trials at all),
            # which is a genuine value rather than a defect — worth recording so
            # the zero-inflation is not mistaken for missingness later.
            "studyPerCap_zero_rows": int((df["studyPerCap"] == 0).sum()),
        },
    }
    return summary


def main() -> None:
    df = load_raw()
    summary = validate(df)

    out = cfg.METRIC_DIR / "data_validation.json"
    out.write_text(json.dumps(summary, indent=2))

    print(f"Loaded {summary['n_rows']:,} rows x {summary['n_cols']} columns")
    print(f"Duplicate rows: {summary['duplicate_rows']}")
    print("\nMissing values:")
    for col, pct in summary["missing_pct"].items():
        print(f"  {col:<28} {summary['missing_counts'][col]:>6}  ({pct}%)")
    print("\nInvalid-value checks:")
    print(json.dumps(summary["invalid_values"], indent=2))
    print(f"\nWrote {out.relative_to(cfg.ROOT)}")


if __name__ == "__main__":
    main()
