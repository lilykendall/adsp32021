"""Tasks 5 & 8 — Evidently AI monitoring setup and drift analysis.

Reference dataset = the **scored original test set** (features + actual target +
model predictions). Current dataset = each scored scenario test set. Using the
untouched test set as reference means any drift Evidently reports is caused by
the controlled perturbation and nothing else.

Two artefacts per scenario:

  * `Report`    — the visual HTML deliverable (data drift, target drift,
                  regression performance).
  * `TestSuite` — the same checks expressed as pass/fail assertions, i.e. what
                  a production monitoring job would actually alert on.

Run standalone (assumes `python -m src.evaluate` has run):

    python -m src.monitoring
"""

from __future__ import annotations

import json

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    RegressionPreset,
    TargetDriftPreset,
)
from evidently.report import Report
from evidently.test_preset import DataDriftTestPreset, RegressionTestPreset
from evidently.test_suite import TestSuite

if __package__ in (None, ""):  # allow `python src/monitoring.py` as well as `-m src.monitoring`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from . import config as cfg
from .preprocess import feature_columns

# Columns the assignment perturbs — pulled out of the drift results for the
# summary table so it is obvious whether monitoring caught each one.
WATCHED_COLUMNS = ["medIncome", "povertyPercent", "AvgHouseholdSize"]


def build_column_mapping(df: pd.DataFrame) -> ColumnMapping:
    """Tell Evidently which column is the target, which is the prediction, and
    which are features. Without this it guesses, and the regression metrics
    silently do not appear."""
    mapping = ColumnMapping()
    mapping.target = cfg.TARGET
    mapping.prediction = "prediction"
    mapping.numerical_features = [
        c for c in feature_columns(df) if c != "prediction"
    ]
    mapping.categorical_features = []
    return mapping


def build_report() -> Report:
    """The monitoring checks configured for this assignment. See README §5.

    - DataDriftPreset      -> per-column input drift (catches Scenarios A/B/C)
    - TargetDriftPreset    -> drift in the target *and* in model predictions,
                              i.e. output/prediction drift — the signal that
                              needs no ground-truth labels
    - DataQualityPreset    -> per-column min/max/mean summaries, which is where
                              the out-of-range values the perturbations create
                              (negative medIncome) become visible
    - RegressionPreset     -> RMSE/MAE/error distribution, reference vs current;
                              retrospective, since it requires ground truth
    """
    return Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset(),
        DataQualityPreset(),
        RegressionPreset(),
    ])


def build_test_suite() -> TestSuite:
    """Pass/fail version of the same checks — the production alerting view."""
    return TestSuite(tests=[DataDriftTestPreset(), RegressionTestPreset()])


def summarise_drift(report_dict: dict) -> dict:
    """Pull the headline drift numbers out of Evidently's `as_dict()` payload."""
    summary = {"n_drifted_columns": None, "share_drifted": None, "columns": {}}

    for metric in report_dict.get("metrics", []):
        if metric.get("metric") == "DatasetDriftMetric":
            res = metric["result"]
            summary["n_drifted_columns"] = res.get("number_of_drifted_columns")
            summary["share_drifted"] = res.get("share_of_drifted_columns")
            summary["dataset_drift"] = res.get("dataset_drift")

        if metric.get("metric") == "DataDriftTable":
            by_col = metric["result"].get("drift_by_columns", {})
            for col, info in by_col.items():
                summary["columns"][col] = {
                    "stattest": info.get("stattest_name"),
                    "drift_score": info.get("drift_score"),
                    "drift_detected": info.get("drift_detected"),
                }

    summary["watched"] = {
        col: summary["columns"].get(col) for col in WATCHED_COLUMNS
    }
    return summary


def run_for_scenario(reference: pd.DataFrame, current: pd.DataFrame,
                     name: str) -> dict:
    """Generate and save the Report + TestSuite for one scenario."""
    mapping = build_column_mapping(reference)

    report = build_report()
    report.run(reference_data=reference, current_data=current,
               column_mapping=mapping)
    report.save_html(str(cfg.REPORT_DIR / f"report_{name}.html"))

    suite = build_test_suite()
    suite.run(reference_data=reference, current_data=current,
              column_mapping=mapping)
    suite.save_html(str(cfg.REPORT_DIR / f"tests_{name}.html"))

    suite_dict = suite.as_dict()
    return {
        "drift": summarise_drift(report.as_dict()),
        "tests": {
            "summary": suite_dict.get("summary", {}),
            "failed": [
                t["name"] for t in suite_dict.get("tests", [])
                if t.get("status") == "FAIL"
            ],
        },
    }


def main() -> None:
    reference = pd.read_csv(cfg.DATA_DIR / "scored_original.csv")
    results = {}

    for name in cfg.SCENARIOS:
        current = pd.read_csv(cfg.DATA_DIR / f"scored_{name}.csv")
        results[name] = run_for_scenario(reference, current, name)
        d = results[name]["drift"]
        print(f"{name:<14} drifted columns: {d['n_drifted_columns']} "
              f"(share {d['share_drifted']}) | failed tests: "
              f"{len(results[name]['tests']['failed'])}")

    (cfg.METRIC_DIR / "drift_summary.json").write_text(json.dumps(results, indent=2))

    # Flat table: did monitoring catch each perturbed column, per scenario?
    rows = []
    for name, res in results.items():
        row = {"dataset": name}
        for col in WATCHED_COLUMNS:
            info = res["drift"]["watched"].get(col) or {}
            row[f"{col}_drift"] = info.get("drift_detected")
            row[f"{col}_score"] = info.get("drift_score")
        row["n_drifted_columns"] = res["drift"]["n_drifted_columns"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(cfg.METRIC_DIR / "drift_by_column.csv", index=False)

    print(f"\nHTML reports -> {cfg.REPORT_DIR.relative_to(cfg.ROOT)}/")
    print(f"Summary      -> artifacts/metrics/drift_summary.json, drift_by_column.csv")


if __name__ == "__main__":
    main()
