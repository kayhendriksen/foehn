"""Ingest MeteoSwiss Parquet files from a Unity Catalog Volume into Delta tables.

Run by the Databricks job after foehn finishes downloading.
Expects parquet files at: /Volumes/<catalog>/<schema>/<volume>/parquet/<collection>/

Usage (local testing with spark-submit):
    spark-submit scripts/ingest_delta.py --catalog main --schema meteoswiss --volume landing
"""

from __future__ import annotations

import argparse

from pyspark.sql import SparkSession

from foehn.collections import COLLECTIONS, GRIB2_COLLECTIONS, NETCDF_COLLECTIONS

# All collections that produce Parquet output (excludes binary/grid collections).
# climate_normals is added manually — it comes from the opendata.swiss ZIP, not STAC.
TABULAR_COLLECTIONS = [
    key for key in COLLECTIONS if key not in GRIB2_COLLECTIONS | NETCDF_COLLECTIONS
] + ["climate_normals"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest MeteoSwiss Parquet → Delta tables.")
    parser.add_argument("--catalog", default="main")
    parser.add_argument("--schema", default="meteoswiss")
    parser.add_argument("--volume", default="landing")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("foehn-ingest").getOrCreate()
    spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "true")

    # Ensure catalog / schema / volume exist (idempotent)
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {args.catalog}")
    spark.sql(f"USE CATALOG {args.catalog}")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {args.catalog}.{args.schema}")
    spark.sql(
        f"CREATE VOLUME IF NOT EXISTS {args.catalog}.{args.schema}.{args.volume}"
    )

    parquet_base = f"/Volumes/{args.catalog}/{args.schema}/{args.volume}/parquet"

    succeeded = 0
    skipped = 0
    for key in TABULAR_COLLECTIONS:
        path = f"{parquet_base}/{key}"
        table = f"{args.catalog}.{args.schema}.{key}"
        try:
            df = spark.read.parquet(path)
            (
                df.write
                .mode("overwrite")
                .option("mergeSchema", "true")
                .saveAsTable(table)
            )
            print(f"  OK  {key:25s} → {table}", flush=True)
            succeeded += 1
        except Exception as e:
            print(f"  --  {key:25s}   skipped ({e})", flush=True)
            skipped += 1

    print(f"\nDone — {succeeded} tables written, {skipped} skipped.")


if __name__ == "__main__":
    main()
