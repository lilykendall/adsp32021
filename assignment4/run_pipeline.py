"""End-to-end driver for Assignment #4.

Runs the whole workflow with no manual intervention:

    python run_pipeline.py

Stages (each is also runnable on its own via `python -m src.<module>`):

    1. src.data        validate the raw dataset
    2. src.preprocess  clean + 80/20 split      -> data/train.csv, data/test_original.csv
    3. src.model       train the regressor      -> artifacts/model.joblib
    4. src.scenarios   build A / A+B / A+B+C    -> data/test_scenario_*.csv
    5. src.validation  pre-inference input gate -> artifacts/metrics/input_validation.*
    6. src.evaluate    score all four datasets  -> artifacts/metrics/, artifacts/figures/
    7. src.compare     linear vs random forest  -> artifacts/metrics/model_comparison.*
    8. src.monitoring  Evidently reports        -> artifacts/reports/
"""

from __future__ import annotations

import time

from src import (
    compare,
    data,
    evaluate,
    model,
    monitoring,
    preprocess,
    scenarios,
    validation,
)

STAGES = [
    ("1. Data loading & validation", data.main),
    ("2. Preprocessing & train/test split", preprocess.main),
    ("3. Baseline model training (Linear Regression)", model.main),
    ("4. Scenario dataset creation", scenarios.main),
    ("5. Pre-inference input validation gate", validation.main),
    ("6. Evaluation (original + 3 scenarios)", evaluate.main),
    ("7. Model comparison (linear vs random forest)", compare.main),
    ("8. Evidently monitoring", monitoring.main),
]


def main() -> None:
    started = time.time()
    for title, fn in STAGES:
        print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
        fn()
    print(f"\n{'=' * 72}")
    print(f"Pipeline complete in {time.time() - started:.1f}s")
    print("Open artifacts/reports/report_scenario_abc.html to see the drift report.")


if __name__ == "__main__":
    main()
