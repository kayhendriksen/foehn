"""Ingest MeteoSwiss Parquet files from a Unity Catalog Volume into Delta tables.

Run by the Databricks job after foehn finishes downloading.
Expects parquet files at: /Volumes/<catalog>/<schema>/<volume>/parquet/<collection>/

Usage (local testing with spark-submit):
    spark-submit scripts/ingest_delta.py --catalog main --schema meteoswiss --volume landing
"""

from __future__ import annotations

import argparse
import re

from pyspark.errors import AnalysisException
from pyspark.sql import SparkSession

from foehn.collections import COLLECTIONS, GRIB2_COLLECTIONS, NETCDF_COLLECTIONS

# All collections that produce Parquet output (excludes binary/grid collections).
# climate_normals is added manually — it comes from the opendata.swiss ZIP, not STAC.
TABULAR_COLLECTIONS = [key for key in COLLECTIONS if key not in GRIB2_COLLECTIONS | NETCDF_COLLECTIONS] + [
    "climate_normals"
]


_IDENTIFIER_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")


def _validate_identifier(value: str, label: str) -> str:
    if not _IDENTIFIER_RE.match(value):
        raise ValueError(f"Invalid {label} {value!r} — only alphanumerics, underscores, and hyphens are allowed")
    return f"`{value}`"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest MeteoSwiss Parquet → Delta tables.")
    parser.add_argument("--catalog", default="main")
    parser.add_argument("--schema", default="meteoswiss")
    parser.add_argument("--volume", default="landing")
    args = parser.parse_args()

    catalog = _validate_identifier(args.catalog, "catalog")
    schema = _validate_identifier(args.schema, "schema")
    volume = _validate_identifier(args.volume, "volume")

    spark = SparkSession.builder.appName("foehn-ingest").getOrCreate()
    spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "true")

    # Ensure catalog / schema / volume exist (idempotent)
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
    spark.sql(f"USE CATALOG {catalog}")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volume}")

    parquet_base = f"/Volumes/{args.catalog}/{args.schema}/{args.volume}/parquet"

    succeeded = 0
    skipped = 0
    for key in TABULAR_COLLECTIONS:
        path = f"{parquet_base}/{key}"
        table = f"{catalog}.{schema}.{key}"
        try:
            df = spark.read.parquet(path)
            (df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(table))
            print(f"  OK  {key:25s} → {table}", flush=True)
            succeeded += 1
        except AnalysisException as e:
            error_class = getattr(e, "getErrorClass", lambda: None)() or "AnalysisException"
            print(f"  --  {key:25s}   skipped ({error_class})", flush=True)
            skipped += 1

    print(f"\nDone — {succeeded} tables written, {skipped} skipped.")


if __name__ == "__main__":
    main()
