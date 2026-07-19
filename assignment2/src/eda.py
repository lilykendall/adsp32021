"""Lightweight exploratory data analysis on the cleaned feature table.

Produces a handful of plots in artifacts/ that document the modelling data:
the target distribution, feature correlations and missingness before cleaning.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config, preprocess


def run() -> None:
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)

    raw = preprocess.ingest()
    clean = preprocess.engineer(preprocess.clean(raw))

    # 1. Missingness in the raw data (top columns).
    miss = (raw.isna().mean().sort_values(ascending=False) * 100).head(15)
    fig, ax = plt.subplots(figsize=(7, 4))
    miss.iloc[::-1].plot.barh(ax=ax, color="#C44E52")
    ax.set_xlabel("% missing (raw)")
    ax.set_title("Missingness by column (raw athletes.csv)")
    fig.tight_layout()
    fig.savefig(os.path.join(config.ARTIFACTS_DIR, "eda_missing.png"), dpi=110)
    plt.close(fig)

    # 2. Target distribution after cleaning.
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(clean[config.TARGET], bins=50, ax=ax, color="#4C72B0")
    ax.set_title("Distribution of total_lift (cleaned)")
    fig.tight_layout()
    fig.savefig(os.path.join(config.ARTIFACTS_DIR, "eda_target_dist.png"), dpi=110)
    plt.close(fig)

    # 3. Correlation heatmap over engineered numeric features + target.
    cols = config.FEATURE_VERSIONS["v2"] + [config.TARGET]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(clean[cols].corr(), annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Feature correlations")
    fig.tight_layout()
    fig.savefig(os.path.join(config.ARTIFACTS_DIR, "eda_correlation.png"), dpi=110)
    plt.close(fig)

    print(f"[eda] rows after cleaning: {len(clean):,}")
    print(f"[eda] target mean={clean[config.TARGET].mean():.1f} "
          f"std={clean[config.TARGET].std():.1f}")
    print("[eda] wrote 3 plots to artifacts/")


if __name__ == "__main__":
    run()
