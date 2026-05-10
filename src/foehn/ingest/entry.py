"""Public ``foehn.ingest.ingest`` entry point.

Auto-derives the inputs the daily Databricks job would otherwise pass
explicitly: ``SparkSession``, bronze path, dataset list, historical
chunking, Unity Catalog DDL.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from foehn.ingest._grouping import _validate_identifier
from foehn.ingest.pipeline import (
    DEFAULT_CHUNK_SIZE,
    RADAR_COLLECTIONS,
    TABULAR_COLLECTIONS,
    run_radar,
    run_tabular,
)
from foehn.ingest.ports import BinaryFileIndex, DeltaSink

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def ingest(
    bronze: Path | str | None = None,
    *,
    catalog: str = "main",
    schema: str = "meteoswiss",
    volume: str = "landing",
    datasets: Iterable[str] | None = None,
    historical: bool | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    spark: SparkSession | None = None,
    sink: DeltaSink | None = None,
    index: BinaryFileIndex | None = None,
) -> tuple[int, int]:
    """Ingest MeteoSwiss bronze data into Delta tables.

    The 90 % case is one line: ``foehn.ingest.ingest()``. Defaults match
    the daily Databricks job.

    Auto-derives:

    - ``SparkSession`` via ``getActiveSession()``, falling back to a fresh
      ``SparkSession.builder`` build.
    - ``bronze`` path as ``/Volumes/{catalog}/{schema}/{volume}/bronze``.
    - ``datasets`` by globbing ``bronze/*`` and intersecting with the
      known tabular and radar dataset keys.
    - ``historical`` from the presence of any ``*_historical*.csv`` file
      under a tabular dataset directory.
    - Unity Catalog DDL (``CREATE CATALOG/SCHEMA/VOLUME IF NOT EXISTS``)
      when ``DATABRICKS_RUNTIME_VERSION`` is set in the environment.

    Tests inject ``sink`` and/or ``index`` to bypass Spark construction.
    Returns ``(succeeded, skipped)`` totals across every dispatched dataset.
    """
    cat = _validate_identifier(catalog, "catalog")
    sch = _validate_identifier(schema, "schema")
    vol = _validate_identifier(volume, "volume")

    bronze_base = Path(bronze) if bronze is not None else Path(f"/Volumes/{catalog}/{schema}/{volume}/bronze")

    if sink is None or index is None:
        if spark is None:
            spark = _get_or_create_spark()
        spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "true")
        if "DATABRICKS_RUNTIME_VERSION" in os.environ:
            spark.sql(f"CREATE CATALOG IF NOT EXISTS {cat}")
            spark.sql(f"USE CATALOG {cat}")
            spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cat}.{sch}")
            spark.sql(f"CREATE VOLUME IF NOT EXISTS {cat}.{sch}.{vol}")
        from foehn.ingest.adapters.spark_delta import SparkBinaryFileIndex, SparkDeltaSink

        if sink is None:
            sink = SparkDeltaSink(spark)
        if index is None:
            index = SparkBinaryFileIndex(spark)

    keys = list(datasets) if datasets is not None else _discover_datasets(bronze_base)

    tabular_set = set(TABULAR_COLLECTIONS)
    radar_set = set(RADAR_COLLECTIONS)
    unknown = [k for k in keys if k not in tabular_set and k not in radar_set]
    if unknown:
        raise ValueError(f"Unknown dataset(s): {unknown}")

    tabular_keys = [k for k in keys if k in tabular_set]
    radar_keys = [k for k in keys if k in radar_set]

    if historical is None:
        historical = _detect_historical(bronze_base, tabular_keys)

    total_ok = 0
    total_skip = 0

    if tabular_keys:
        ok, skip = run_tabular(
            bronze_base,
            cat,
            sch,
            sink,
            keys=tabular_keys,
            chunked=historical,
            chunk_size=chunk_size,
        )
        total_ok += ok
        total_skip += skip

    if radar_keys:
        ok, skip = run_radar(bronze_base, cat, sch, index, keys=radar_keys)
        total_ok += ok
        total_skip += skip

    return total_ok, total_skip


def _discover_datasets(bronze_base: Path) -> list[str]:
    """Sorted list of *tabular* dataset keys present under ``bronze_base``.

    Auto-discovery defaults to tabular-only — radar callers must pass
    ``datasets=["radar_precip", ...]`` explicitly. Daily ingest stays cheap;
    radar runs on its own 5-min schedule and would re-scan thousands of
    HDF5 files unnecessarily if the daily job picked them up.
    """
    if not bronze_base.exists():
        return []
    tabular = set(TABULAR_COLLECTIONS)
    return sorted(d.name for d in bronze_base.iterdir() if d.is_dir() and d.name in tabular)


def _detect_historical(bronze_base: Path, tabular_keys: Iterable[str]) -> bool:
    """True iff any tabular dataset directory contains a ``*_historical*.csv`` file."""
    for key in tabular_keys:
        d = bronze_base / key
        if d.exists() and any(d.glob("*_historical*.csv")):
            return True
    return False


def _get_or_create_spark() -> Any:
    from pyspark.sql import SparkSession

    active = SparkSession.getActiveSession()
    if active is not None:
        return active
    return SparkSession.builder.appName("foehn-ingest").getOrCreate()
