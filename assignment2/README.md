# ADSP 31021 — Assignment 2: Feature Store & Reproducible MLOps Pipeline

An end-to-end, reproducible machine-learning workflow over the CrossFit
`athletes.csv` dataset. The pipeline integrates a **feature store (Feast)**,
**experiment tracking (MLflow)**, **feature versioning**, and a **4-run
experiment matrix** built from two feature versions × two hyperparameter
configurations of a single algorithm.

**Task:** regression — predict an athlete's `total_lift`
(`deadlift + clean&jerk + snatch + back-squat`, in lbs) from their profile.

---

## 1. Results summary

Four experiments, same algorithm (`RandomForestRegressor`), tracked in MLflow.
Sorted by RMSE (lower is better):

| Experiment | Feature version | Hyperparameters      | RMSE   | MAE    | R²     |
|------------|-----------------|----------------------|--------|--------|--------|
| v2_hp_a    | v2 (6 features) | n=100, max_depth=8   | **159.96** | **122.63** | **0.669** |
| v1_hp_a    | v1 (4 features) | n=100, max_depth=8   | 165.46 | 128.58 | 0.647  |
| v2_hp_b    | v2 (6 features) | n=300, max_depth=20  | 173.11 | 132.83 | 0.612  |
| v1_hp_b    | v1 (4 features) | n=300, max_depth=20  | 182.86 | 142.00 | 0.569  |

**Takeaways**
- **v2 > v1** at every hyperparameter setting: the engineered training-behaviour
  features (`howlong_enc`, `schedule_enc`) add real signal.
- **hp_a > hp_b**: the shallow, smaller forest (`max_depth=8`) generalises
  better; the deep `max_depth=20` forest overfits (worse held-out RMSE despite
  more trees). This is exactly the kind of finding experiment tracking exists to
  surface.
- Best model: **v2_hp_a**, R² ≈ 0.67.

Full numbers: [`artifacts/experiment_summary.csv`](artifacts/experiment_summary.csv).
Comparison chart: [`artifacts/experiment_comparison.png`](artifacts/experiment_comparison.png).

---

## 2. Tooling choices & rationale

| Concern | Tool | Why |
|---------|------|-----|
| **MLOps platform / experiment tracking** | **MLflow** 2.19 | Lightweight, file-backed (no server needed), first-class params/metrics/tags/artifacts/model logging, and a built-in UI for comparing runs. Ideal for a small reproducible course project. |
| **Feature store** | **Feast** 0.40 | The de-facto open-source feature store. Runs fully local (file offline store + SQLite online store) so it needs no cloud services, yet demonstrates the real store→retrieve→serve lifecycle and versioning via **feature services**. |
| **Model** | `RandomForestRegressor` (scikit-learn) | Assignment requires one algorithm across all experiments; RF is a strong, low-tuning tabular baseline and matches Assignment 1 for comparability. |

Both platforms were chosen because they run **from a clean environment with no
external infrastructure** — the whole thing is reproducible on a laptop.

---

## 3. Repository layout

```
assignment2/
├── athletes.csv                 # raw dataset (provided)
├── requirements.txt             # pinned dependencies (Python 3.9)
├── run_pipeline.sh              # one-command end-to-end run
├── run_experiments.py           # the 2x2 experiment matrix (MLflow tracking)
├── demo_feature_store.py        # offline + online feature retrieval demo
├── src/
│   ├── config.py                # single source of truth: paths, seeds, versions, HPs
│   ├── preprocess.py            # ingest → clean → feature-engineer → Feast parquet
│   ├── pipeline.py              # feature retrieval + train + evaluate stages
│   └── eda.py                   # exploratory plots
├── feature_repo/                # Feast feature store
│   ├── feature_store.yaml       # local file/SQLite provider config
│   ├── definitions.py           # entity, feature views, v1/v2 feature services
│   └── data/                    # generated: parquet source + registry + online store
├── artifacts/                   # metrics CSV, comparison + diagnostic plots (committed as evidence)
└── mlruns/                      # MLflow tracking store (generated)
```

---

## 4. Setup & reproduction

Requires Python 3.9+.

```bash
cd assignment2

# 1. Create an isolated environment and install pinned deps
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

# 2. Run everything end-to-end (EDA → preprocess → feast apply → 4 experiments → demo)
bash run_pipeline.sh
```

Or run the stages individually:

```bash
.venv/bin/python -m src.eda            # exploratory plots → artifacts/
.venv/bin/python -m src.preprocess     # build feature_repo/data/athlete_features.parquet
cd feature_repo && ../.venv/bin/feast apply && cd ..   # register feature definitions
.venv/bin/python run_experiments.py    # run + track the 4 experiments
.venv/bin/python demo_feature_store.py # demonstrate offline + online retrieval
```

**Inspect the tracked experiments in the MLflow UI:**

```bash
.venv/bin/mlflow ui --backend-store-uri ./mlruns
# open http://localhost:5000  →  experiment "crossfit_total_lift"
```

The pipeline is deterministic (`RANDOM_SEED = 42`); re-running reproduces the
metrics in the table above.

---

## 5. Dataset assumptions & preprocessing decisions

Cleaning is inherited from Assignment 1 so the two assignments stay comparable.
Documented in [`src/preprocess.py`](src/preprocess.py):

- **Target** `total_lift = deadlift + candj + snatch + backsq`; rows missing any
  lift are dropped.
- **Row filters** remove physiologically implausible records: `age ≥ 18`;
  `48 < height < 96` in; `weight < 1500` lbs; lift ceilings per gender
  (e.g. male deadlift ≤ 1105, female ≤ 636); positive lift values within realistic
  world-record-scaled caps.
- **Dropped columns**: benchmark WOD times and identity columns
  (`name`, `team`, `affiliate`, `fran`, `helen`, …) that are either leakage or
  irrelevant to a strength model.
- **`"Decline to answer|"`** survey responses are treated as missing and dropped.
- **Entity/timestamp for Feast**: `athlete_id` is the entity key; the dataset is a
  static snapshot with no natural event time, so every row gets one constant
  `event_timestamp` (2024-01-01). The point-in-time join is therefore trivial but
  demonstrated correctly.
- **Result:** 30,190 clean athletes; mean `total_lift` ≈ 1018 lbs (σ ≈ 278).

EDA plots: `artifacts/eda_missing.png`, `eda_target_dist.png`, `eda_correlation.png`.

---

## 6. Feature store integration & versioning

Features are defined once in [`feature_repo/definitions.py`](feature_repo/definitions.py)
and served through Feast. Versioning uses **feature services** — Feast's idiomatic
unit of versioning — so each model run references an immutable, named feature bundle
that is recorded in the registry.

| Version | Feast feature service | Features | Rationale |
|---------|----------------------|----------|-----------|
| **v1** | `athlete_service_v1` | `age, weight, height, gender_enc` | Physical / demographic baseline. |
| **v2** | `athlete_service_v2` | v1 **+** `howlong_enc, schedule_enc` | Adds engineered training-engagement signals (how long they've trained, weekly schedule intensity). |

The features live in two feature views (`athlete_physical`, `athlete_training`) so
v2 composes v1's view rather than duplicating column definitions.

`demo_feature_store.py` demonstrates the full lifecycle:
1. **Store** — `feast apply` registers definitions; `materialize` loads features into
   the online SQLite store.
2. **Retrieve (offline)** — `get_historical_features(...)` with a feature service
   builds the training set (this is what the training pipeline calls).
3. **Retrieve (online)** — `get_online_features(...)` fetches a single athlete's
   features by entity key, the serving-time path.

The training code (`src/pipeline.py`) **never reads raw columns directly** — it
always asks Feast for the named feature service, so the feature version used by each
model is explicit and tracked.

---

## 7. Experiment design

Per the assignment, the four experiments vary only feature version and
hyperparameters, holding the algorithm fixed:

|             | **hp_a** (n=100, depth=8) | **hp_b** (n=300, depth=20) |
|-------------|---------------------------|----------------------------|
| **v1**      | v1_hp_a                    | v1_hp_b                     |
| **v2**      | v2_hp_a                    | v2_hp_b                     |

Each MLflow run records: the feature version + feature-service name (tags), the
hyperparameters and feature list (params), RMSE/MAE/R²/train-test sizes (metrics),
and per-run diagnostic plots plus the serialized model (artifacts). No AutoML or
automated algorithm selection is used.

---

## 8. Deliverable checklist

- ✅ Working code (this repo) & one-command run (`run_pipeline.sh`)
- ✅ README with setup + execution instructions (this file)
- ✅ Experiment comparison summary (`artifacts/experiment_summary.csv` + chart)
- ✅ Evaluation metrics & visualizations (`artifacts/*.png`)
- ✅ Feature versioning evidence (two Feast feature services; §6)
- ✅ Experiment tracking evidence (MLflow `mlruns/`, 4 runs)
- ✅ Reproducibility (pinned `requirements.txt`, fixed seed, clean-env run)
