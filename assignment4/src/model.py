"""Task 3 — Baseline model development.

The model is a single sklearn `Pipeline` so that imputation + scaling are fit on
training data only and travel with the estimator when it is applied to the
perturbed scenario datasets.

The primary (deployed and monitored) model is **Linear Regression**. A Random
Forest is trained separately by `src/compare.py` as a documented benchmark; it
is never saved over `artifacts/model.joblib` and never monitored.

Run standalone (assumes `python -m src.preprocess` has run):

    python -m src.model
"""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):  # allow `python src/model.py` as well as `-m src.model`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from . import config as cfg
from .preprocess import feature_columns


def make_estimator(name: str) -> BaseEstimator:
    """Construct one of the candidate estimators by name.

    `linear` is the assignment's chosen model. `random_forest` exists only so
    `src/compare.py` can quantify what the linear assumption costs in accuracy
    and how differently the two extrapolate under the scenario perturbations.
    """
    if name == "linear":
        return LinearRegression()
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=cfg.RANDOM_SEED,
        )
    raise ValueError(f"unknown model {name!r}")


def build_pipeline(numeric_features: list[str], model: str | None = None) -> Pipeline:
    """Preprocessing + estimator in one fitted object.

    Median imputation and standard scaling both learn statistics from the data,
    so they live *inside* the pipeline and are therefore fit on training rows
    only. Scaling is not strictly required by OLS, but it puts every coefficient
    on a comparable footing, which is what makes the coefficient table in
    README §3 readable.
    """
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        [("numeric", numeric_pipe, numeric_features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    estimator = make_estimator(model or cfg.PRIMARY_MODEL)
    return Pipeline([("prep", preprocessor), ("model", estimator)])


def train(train_df: pd.DataFrame, model: str | None = None) -> Pipeline:
    features = feature_columns(train_df)
    pipe = build_pipeline(features, model=model)
    pipe.fit(train_df[features], train_df[cfg.TARGET])
    return pipe


def load_model() -> Pipeline:
    return joblib.load(cfg.MODEL_PATH)


def coefficient_table(pipe: Pipeline, features: list[str]) -> pd.DataFrame:
    """Standardised and raw-unit coefficients for a fitted linear pipeline.

    The pipeline standardises features before fitting, so `coef_` is the effect
    of a one-standard-deviation move (comparable across columns). Dividing by
    the scaler's per-column scale recovers the effect of one *raw* unit — one
    dollar of `medIncome`, one point of `povertyPercent` — which is what makes
    the scenario shifts interpretable.
    """
    estimator = pipe.named_steps["model"]
    if not hasattr(estimator, "coef_"):
        raise TypeError(f"{type(estimator).__name__} has no coefficients")

    scaler = pipe.named_steps["prep"].named_transformers_["numeric"].named_steps["scale"]
    std_coef = pd.Series(estimator.coef_, index=features)
    raw_coef = std_coef / pd.Series(scaler.scale_, index=features)

    table = pd.DataFrame(
        {"std_coef": std_coef, "raw_unit_coef": raw_coef, "abs_std_coef": std_coef.abs()}
    ).sort_values("abs_std_coef", ascending=False)
    table.index.name = "feature"
    return table.drop(columns="abs_std_coef")


def main() -> None:
    train_df = pd.read_csv(cfg.DATA_DIR / "train.csv")
    features = feature_columns(train_df)
    pipe = train(train_df)
    joblib.dump(pipe, cfg.MODEL_PATH)

    estimator = pipe.named_steps["model"]
    print(f"Trained {type(estimator).__name__} on {len(train_df):,} rows, "
          f"{len(features)} features -> {cfg.MODEL_PATH.relative_to(cfg.ROOT)}")

    if hasattr(estimator, "coef_"):
        coefs = coefficient_table(pipe, features)
        coefs.to_csv(cfg.METRIC_DIR / "linear_coefficients.csv")

        # The per-raw-unit coefficient on each perturbed column is exactly the
        # per-row prediction shift that scenario will produce — a linear model
        # lets us predict the damage before scoring anything.
        print(f"\nIntercept: {estimator.intercept_:.3f}")
        print("\nPredicted effect of each scenario shift on every prediction:")
        for col, delta in {**cfg.SCENARIO_A, **cfg.SCENARIO_B, **cfg.SCENARIO_C}.items():
            effect = coefs.loc[col, "raw_unit_coef"] * delta
            print(f"  {col:<18} {delta:+11,g} x {coefs.loc[col, 'raw_unit_coef']:+.6f}"
                  f"  ->  {effect:+7.2f} deaths/100k")

        print("\nTop 8 features by |standardised coefficient|:")
        print(coefs.head(8).round(4).to_string())
        print(f"\nWrote {(cfg.METRIC_DIR / 'linear_coefficients.csv').relative_to(cfg.ROOT)}")


if __name__ == "__main__":
    main()
