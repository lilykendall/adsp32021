#!/usr/bin/env bash
# End-to-end pipeline: clean env -> preprocess -> register feature store ->
# run the 4 experiments. Run from the assignment2/ directory.
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
FEAST="${FEAST:-$ROOT/.venv/bin/feast}"

echo ">>> [1/4] EDA + preprocessing (build Feast source parquet)"
$PY -m src.eda
$PY -m src.preprocess

echo ">>> [2/4] Register feature definitions with Feast"
( cd feature_repo && "$FEAST" apply )

echo ">>> [3/4] Run the 2x2 experiment matrix (tracked in MLflow)"
$PY run_experiments.py

echo ">>> [4/4] Demonstrate feature-store lifecycle (offline + online retrieval)"
$PY demo_feature_store.py

echo
echo "Done. Inspect results with:"
echo "  .venv/bin/mlflow ui --backend-store-uri ./mlruns   # then open http://localhost:5000"
echo "  open artifacts/experiment_comparison.png"
