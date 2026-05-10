"""Ingest MeteoSwiss bronze data into Delta tables on Databricks.

Boot layer: parses CLI args, builds a SparkSession, runs Unity Catalog
DDL when running on Databricks, constructs the production
:class:`SparkDeltaSink` / :class:`SparkBinaryFileIndex` adapters, and
dispatches to the pipeline functions in :mod:`foehn.ingest.pipeline`.

Run by the Databricks job after foehn finishes downloading.

Usage (local testing with spark-submit):
    spark-submit scripts/ingest_delta.py --catalog main --schema meteoswiss --volume landing
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from pyspark.sql import SparkSession

from foehn.ingest._grouping import _validate_identifier
from foehn.ingest.adapters.spark_delta import SparkBinaryFileIndex, SparkDeltaSink
from foehn.ingest.pipeline import (
    DEFAULT_CHUNK_SIZE,
    RADAR_COLLECTIONS,
    run_radar,
    run_tabular,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest MeteoSwiss CSVs → Delta tables via Polars + Spark.")
    parser.add_argument("--catalog", default="main")
    parser.add_argument("--schema", default="meteoswiss")
    parser.add_argument("--volume", default="landing")
    parser.add_argument(
        "--historical",
        action="store_true",
        help="Enable chunked writes for large collections (SMN, etc.)",
    )
    parser.add_argument(
        "--radar",
        nargs="*",
        default=None,
        metavar="COLLECTION",
        help="Only index radar HDF5 files into Delta catalogs. Pass one or more "
        "collection keys (radar_precip, radar_hail) to restrict; bare --radar ingests all.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Station files per chunk for large historical collections (default: {DEFAULT_CHUNK_SIZE})",
    )
    args = parser.parse_args()

    cat = _validate_identifier(args.catalog, "catalog")
    sch = _validate_identifier(args.schema, "schema")
    vol = _validate_identifier(args.volume, "volume")

    spark = SparkSession.builder.appName("foehn-ingest").getOrCreate()
    spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "true")

    # Unity Catalog DDL — only available on Databricks, skip for local spark-submit.
    on_databricks = "DATABRICKS_RUNTIME_VERSION" in os.environ
    if on_databricks:
        spark.sql(f"CREATE CATALOG IF NOT EXISTS {cat}")
        spark.sql(f"USE CATALOG {cat}")
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cat}.{sch}")
        spark.sql(f"CREATE VOLUME IF NOT EXISTS {cat}.{sch}.{vol}")

    sink = SparkDeltaSink(spark)
    index = SparkBinaryFileIndex(spark)
    bronze_base = Path(f"/Volumes/{args.catalog}/{args.schema}/{args.volume}/bronze")

    if args.radar is not None:
        keys = tuple(args.radar) if args.radar else RADAR_COLLECTIONS
        invalid = [k for k in keys if k not in RADAR_COLLECTIONS]
        if invalid:
            raise SystemExit(f"Unknown radar collection(s): {invalid}. Valid: {list(RADAR_COLLECTIONS)}")
        ok, skip = run_radar(bronze_base, cat, sch, index, keys=keys)
        print(f"\nDone — {ok} tables written, {skip} skipped.")
        return

    ok, skip = run_tabular(
        bronze_base,
        cat,
        sch,
        sink,
        chunked=args.historical,
        chunk_size=args.chunk_size,
    )
    print(f"\nDone — {ok} tables written, {skip} skipped.")


if __name__ == "__main__":
    main()
