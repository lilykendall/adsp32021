"""Demonstrate the full feature-store lifecycle: store -> retrieve -> use.

Runs three things end to end:
  1. Offline (historical) retrieval for both feature versions -- the exact call
     the training pipeline uses to build its training set.
  2. Materialisation of features into the online (SQLite) store.
  3. Online retrieval of a single athlete's features by entity key, the way a
     real-time serving path would fetch them at inference time.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from src import config, pipeline


def main() -> None:
    store = pipeline.get_store()

    print("=" * 70)
    print("1) OFFLINE RETRIEVAL (historical features for training)")
    for version in config.FEATURE_VERSIONS:
        df = pipeline.load_features(store, version)
        cols = config.FEATURE_VERSIONS[version]
        print(f"  {config.FEATURE_SERVICE_NAMES[version]}: "
              f"{len(df):,} rows, features={cols}")

    print("=" * 70)
    print("2) MATERIALISE features into the online store")
    store.materialize_incremental(end_date=datetime.utcnow())

    print("=" * 70)
    print("3) ONLINE RETRIEVAL (serving-time lookup by entity key)")
    sample_ids = pd.read_parquet(config.FEATURES_PARQUET)[config.ENTITY_KEY].head(3).tolist()
    online = store.get_online_features(
        features=store.get_feature_service("athlete_service_v2"),
        entity_rows=[{"athlete_id": i} for i in sample_ids],
    ).to_df()
    print(online.to_string(index=False))


if __name__ == "__main__":
    main()
