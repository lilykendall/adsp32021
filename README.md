# ADSP 31021 — Machine Learning Operations

Coursework for **ADSP 31021 (Machine Learning Operations)**, MS in Applied Data
Science, University of Chicago.

**Student:** Lily Kendall

Assignments 1–3 build reproducible ML workflows over the **CrossFit Open athletes
dataset** (`athletes.csv`), predicting `total_lift`
(deadlift + clean-and-jerk + snatch + back-squat, in lbs) from athlete profiles.
Assignment 4 uses the instructor-provided **county cancer mortality dataset**
(`cancer_reg.csv`), predicting `TARGET_deathRate`.

## Assignments

| # | Topic | Key tooling | Folder |
|---|-------|-------------|--------|
| 1 | Data Versioning | DVC, scikit-learn / XGBoost, Ruff | [`assignment1/`](assignment1/) |
| 2 | Feature Store & Reproducible MLOps Pipeline | Feast (feature store), MLflow (experiment tracking), scikit-learn | [`assignment2/`](assignment2/) |
| 3 | AutoML | PyCaret + MLflow (chosen platform), H2O AutoML (required repeat) | [`assignment3/`](assignment3/) |
| 4 | Model Monitoring | Evidently AI (drift + regression reports), scikit-learn | [`assignment4/`](assignment4/) |

### Assignment 1 — Data Versioning
A Jupyter notebook that versions two dataset iterations (raw v1, processed v2)
with **DVC**, performs EDA, and trains a model on the CrossFit dataset.
See [`assignment1/`](assignment1/).

### Assignment 2 — Feature Store & Reproducible MLOps Pipeline
An end-to-end, modular pipeline integrating a **Feast** feature store (two
versioned feature services) and **MLflow** experiment tracking. Runs a 2×2 matrix
of two feature versions × two RandomForest hyperparameter configurations, all
reproducible from a clean environment.
Full setup and results in [`assignment2/README.md`](assignment2/README.md).

### Assignment 3 — AutoML
Compares automated model selection against the Assignment 1 baseline using
**PyCaret** (chosen MLOps platform, with **MLflow** as its experiment-tracking
backend) and the required **H2O AutoML** repeat — leaderboards, feature
importance, and speed/validation-score tradeoffs for both an all-features run
and a reduced top-3-feature run.
Full setup and results in [`assignment3/README.md`](assignment3/README.md).

### Assignment 4 — Model Monitoring
A post-deployment monitoring workflow on a **different dataset** — county-level
US cancer mortality (`cancer_reg.csv`) — predicting `TARGET_deathRate`. A
**Linear Regression** model (benchmarked against a RandomForest) is scored
against the original test set and three cumulatively perturbed versions of it,
with **Evidently AI** detecting input drift and prediction drift in each
scenario, plus a pre-inference input-validation gate.

The headline result: in Scenario A+B the income and poverty perturbations cancel
almost exactly, so accuracy and prediction drift both report a perfectly healthy
model while 31% of input rows contain negative median incomes — caught only by
per-column input drift and the bounds gate.
Full setup and results in [`assignment4/README.md`](assignment4/README.md).

## Repository layout

```
adsp32021/
├── assignment1/    # Data versioning with DVC
├── assignment2/    # Feature store + MLflow pipeline (see assignment2/README.md)
├── assignment3/    # AutoML: PyCaret+MLflow and H2O (see assignment3/README.md)
└── assignment4/    # Model monitoring with Evidently AI (see assignment4/README.md)
```

Each assignment folder is self-contained with its own `requirements.txt` and
setup instructions.
