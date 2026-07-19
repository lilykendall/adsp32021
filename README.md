# ADSP 31021 — Machine Learning Operations

Coursework for **ADSP 31021 (Machine Learning Operations)**, MS in Applied Data
Science, University of Chicago.

**Student:** Lily Kendall

All assignments build reproducible ML workflows over the **CrossFit Open athletes
dataset** (`athletes.csv`), predicting `total_lift`
(deadlift + clean-and-jerk + snatch + back-squat, in lbs) from athlete profiles.

## Assignments

| # | Topic | Key tooling | Folder |
|---|-------|-------------|--------|
| 1 | Data Versioning | DVC, scikit-learn / XGBoost, Ruff | [`assignment1/`](assignment1/) |
| 2 | Feature Store & Reproducible MLOps Pipeline | Feast (feature store), MLflow (experiment tracking), scikit-learn | [`assignment2/`](assignment2/) |

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

## Repository layout

```
adsp32021/
├── assignment1/    # Data versioning with DVC
└── assignment2/    # Feature store + MLflow pipeline (see assignment2/README.md)
```

Each assignment folder is self-contained with its own `requirements.txt` and
setup instructions.
