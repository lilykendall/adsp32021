"""Pre-inference input validation gate.

This is the safeguard that *should* sit in front of the model in production, and
the one that would have caught Scenario A before any prediction was made. It
answers a question drift detection cannot: not "have these inputs changed?" but
"are these inputs possible at all?"

Two distinct kinds of problem are reported, and the distinction matters:

  * **Hard-bound violation** — the value is physically impossible (a negative
    median income). In production this should *block* scoring for that row.
  * **Out-of-distribution** — the value is possible but lies outside the range
    the model was trained on (`AvgHouseholdSize` of 5.97 when training topped out
    at 3.97). The model is extrapolating; the prediction is not trustworthy, but
    it is not nonsense either. This should *warn*, not block.

The gate runs in **report-only** mode here. It records every violation and
writes the evidence to `artifacts/metrics/`, but deliberately does not drop or
clip any row: the assignment requires the perturbed values be scored exactly as
specified so that Evidently sees the true shift, and dropping rows would leave
the four scenario datasets with different row counts and non-comparable metrics.
`enforce=True` shows what the blocking version would reject.

Run standalone (assumes `python -m src.scenarios` has run):

    python -m src.validation
"""

from __future__ import annotations

import json

import pandas as pd

if __package__ in (None, ""):  # allow `python src/validation.py` as well as `-m src.validation`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from . import config as cfg
from .preprocess import feature_columns


def resolve_bounds(columns: list[str]) -> dict[str, tuple[float | None, float | None]]:
    """Hard bounds for each column: explicit config entries first, then the
    naming convention — anything called `Pct*` or `Percent*` is a percentage and
    must lie in [0, 100]."""
    bounds: dict[str, tuple[float | None, float | None]] = {}
    for col in columns:
        if col in cfg.INPUT_BOUNDS:
            bounds[col] = cfg.INPUT_BOUNDS[col]
        elif col.startswith("Pct") or col.startswith("Percent"):
            bounds[col] = (0.0, 100.0)
    return bounds


def training_ranges(train_df: pd.DataFrame, columns: list[str]) -> dict[str, dict]:
    """The min/max the model actually saw, per column — the reference for
    out-of-distribution detection."""
    return {
        col: {"min": float(train_df[col].min()), "max": float(train_df[col].max())}
        for col in columns
        if col in train_df.columns
    }


def validate_inputs(df: pd.DataFrame, train_ranges: dict[str, dict],
                    name: str = "dataset") -> dict:
    """Check one dataset against hard bounds and against the training ranges."""
    columns = [c for c in feature_columns(df) if c != "prediction"]
    bounds = resolve_bounds(columns)

    violations: dict[str, dict] = {}
    ood: dict[str, dict] = {}
    blocked = pd.Series(False, index=df.index)

    for col in columns:
        series = df[col]

        lo, hi = bounds.get(col, (None, None))
        below = series < lo if lo is not None else pd.Series(False, index=df.index)
        above = series > hi if hi is not None else pd.Series(False, index=df.index)
        bad = (below | above).fillna(False)

        if bad.any():
            blocked |= bad
            violations[col] = {
                "bound_min": lo,
                "bound_max": hi,
                "n_violating": int(bad.sum()),
                "pct_violating": round(100 * float(bad.mean()), 2),
                "observed_min": float(series.min()),
                "observed_max": float(series.max()),
            }

        rng = train_ranges.get(col)
        if rng is not None:
            outside = (
                (series < rng["min"]) | (series > rng["max"])
            ).fillna(False)
            if outside.any():
                ood[col] = {
                    "train_min": rng["min"],
                    "train_max": rng["max"],
                    "n_outside": int(outside.sum()),
                    "pct_outside": round(100 * float(outside.mean()), 2),
                    "observed_min": float(series.min()),
                    "observed_max": float(series.max()),
                    "alert": bool(outside.mean() > cfg.OOD_ALERT_THRESHOLD),
                }

    n_alerting = sum(1 for v in ood.values() if v["alert"])

    # A handful of test rows always fall marginally outside the training range —
    # that is ordinary sampling variation, not drift. Only an excursion affecting
    # more than OOD_ALERT_THRESHOLD of the batch counts as a warning.
    if violations:
        status = "REJECT"
    elif n_alerting:
        status = "WARN"
    else:
        status = "PASS"

    return {
        "dataset": name,
        "n_rows": int(len(df)),
        "n_columns_checked": len(columns),
        "hard_bound_violations": violations,
        "n_rows_blocked": int(blocked.sum()),
        "pct_rows_blocked": round(100 * float(blocked.mean()), 2),
        "out_of_distribution": ood,
        "n_columns_ood_alerting": n_alerting,
        # The verdict a production gate would return for this batch.
        "gate_status": status,
        "enforced": False,  # report-only; see the module docstring
    }


def apply_gate(df: pd.DataFrame, train_ranges: dict[str, dict],
               enforce: bool = False) -> tuple[pd.DataFrame, dict]:
    """Validate `df` and, if `enforce`, return only the rows that passed.

    The pipeline always calls this with `enforce=False`. The parameter exists so
    the blocking behaviour is implemented and demonstrable rather than merely
    described in the README.
    """
    result = validate_inputs(df, train_ranges)
    if not enforce:
        return df, result

    columns = [c for c in feature_columns(df) if c != "prediction"]
    bounds = resolve_bounds(columns)
    keep = pd.Series(True, index=df.index)
    for col, (lo, hi) in bounds.items():
        if lo is not None:
            keep &= (df[col] >= lo) | df[col].isna()
        if hi is not None:
            keep &= (df[col] <= hi) | df[col].isna()

    result["enforced"] = True
    result["n_rows_scored"] = int(keep.sum())
    return df[keep], result


def main() -> None:
    train_df = pd.read_csv(cfg.DATA_DIR / "train.csv")
    columns = feature_columns(train_df)
    train_ranges = training_ranges(train_df, columns)
    (cfg.METRIC_DIR / "training_ranges.json").write_text(
        json.dumps(train_ranges, indent=2)
    )

    results, rows = {}, []
    for name in cfg.SCENARIOS:
        path = cfg.DATA_DIR / (
            "test_original.csv" if name == "original" else f"test_{name}.csv"
        )
        df = pd.read_csv(path)
        result = validate_inputs(df, train_ranges, name=name)
        results[name] = result

        violating = ", ".join(
            f"{c} ({v['n_violating']} rows)"
            for c, v in result["hard_bound_violations"].items()
        ) or "none"
        print(f"{name:<14} gate={result['gate_status']:<7} "
              f"rows failing hard bounds: {result['n_rows_blocked']:>3}/{result['n_rows']} "
              f"| impossible columns: {violating}")

        rows.append({
            "dataset": name,
            "gate_status": result["gate_status"],
            "n_rows": result["n_rows"],
            "n_rows_blocked": result["n_rows_blocked"],
            "pct_rows_blocked": result["pct_rows_blocked"],
            "n_columns_impossible": len(result["hard_bound_violations"]),
            "columns_impossible": ";".join(result["hard_bound_violations"]) or None,
            "n_columns_ood": len(result["out_of_distribution"]),
            "columns_ood_alerting": ";".join(
                c for c, v in result["out_of_distribution"].items() if v["alert"]
            ) or None,
        })

    (cfg.METRIC_DIR / "input_validation.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(rows).to_csv(cfg.METRIC_DIR / "input_validation.csv", index=False)

    print("\nGate runs in report-only mode: no rows were dropped, so the scenario "
          "metrics stay comparable and Evidently sees the full shift.")
    print("Wrote artifacts/metrics/input_validation.json, input_validation.csv, "
          "training_ranges.json")


if __name__ == "__main__":
    main()
