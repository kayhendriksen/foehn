"""Boot layer for the foehn Delta ingestion job.

Parses CLI args and delegates to :func:`foehn.ingest.ingest`, which
acquires the SparkSession, runs Unity Catalog DDL when on Databricks,
constructs the production adapters, and dispatches to the pipeline.

Usage (local testing with spark-submit):
    spark-submit scripts/ingest_delta.py --catalog main --schema meteoswiss --volume landing
"""

from __future__ import annotations

import argparse

from foehn.ingest import ingest


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest MeteoSwiss bronze data into Delta tables.")
    parser.add_argument("--catalog", default="main")
    parser.add_argument("--schema", default="meteoswiss")
    parser.add_argument("--volume", default="landing")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="DATASET",
        help="Dataset keys to ingest (e.g. radar_precip, radar_hail, smn). "
        "Omit to auto-discover every tabular dataset present under bronze.",
    )
    args = parser.parse_args()

    ok, skip = ingest(
        catalog=args.catalog,
        schema=args.schema,
        volume=args.volume,
        datasets=args.datasets,
    )
    print(f"\nDone — {ok} tables written, {skip} skipped.")


if __name__ == "__main__":
    main()
