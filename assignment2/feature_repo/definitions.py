"""Feast feature definitions for the CrossFit athletes project.

Feature versioning strategy
---------------------------
Two feature *services* express the two required feature versions. A feature
service is Feast's idiomatic unit of versioning: it names an immutable bundle of
features that a model consumes, so training runs can reference a version by name
and that reference is recorded in the registry.

    athlete_service_v1  ->  physical / demographic baseline
    athlete_service_v2  ->  v1 + engineered training-behaviour features

The underlying features live in two feature views so v2 can reuse v1's view
rather than duplicating column definitions.
"""
import os
from datetime import timedelta

from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    ValueType,
)
from feast.types import Float32, Int64

# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------
athlete = Entity(
    name="athlete",
    join_keys=["athlete_id"],
    value_type=ValueType.INT64,
    description="A single CrossFit Open competitor.",
)

# ---------------------------------------------------------------------------
# Data source (offline / file store) -- produced by src/preprocess.py
# ---------------------------------------------------------------------------
# Absolute path (resolved from this file) so historical retrieval works no
# matter which directory the pipeline is launched from.
_PARQUET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "athlete_features.parquet"
)
athlete_source = FileSource(
    name="athlete_source",
    path=_PARQUET_PATH,
    timestamp_field="event_timestamp",
)

# ---------------------------------------------------------------------------
# Feature views
# ---------------------------------------------------------------------------
physical_fv = FeatureView(
    name="athlete_physical",
    entities=[athlete],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="age", dtype=Float32),
        Field(name="weight", dtype=Float32),
        Field(name="height", dtype=Float32),
        Field(name="gender_enc", dtype=Int64),
    ],
    source=athlete_source,
    online=True,
    description="Demographic and anthropometric features.",
)

training_fv = FeatureView(
    name="athlete_training",
    entities=[athlete],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="howlong_enc", dtype=Int64),
        Field(name="schedule_enc", dtype=Int64),
    ],
    source=athlete_source,
    online=True,
    description="Engineered training-engagement features.",
)

# ---------------------------------------------------------------------------
# Feature services == feature versions
# ---------------------------------------------------------------------------
athlete_service_v1 = FeatureService(
    name="athlete_service_v1",
    features=[physical_fv],
    description="v1: physical/demographic baseline.",
)

athlete_service_v2 = FeatureService(
    name="athlete_service_v2",
    features=[physical_fv, training_fv],
    description="v2: v1 + training-behaviour features.",
)
