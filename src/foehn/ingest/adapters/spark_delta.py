"""Production adapters that talk to Spark and Delta."""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from foehn.ingest.ports import WriteMode

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


class SparkDeltaSink:
    """``DeltaSink`` backed by a Spark session writing Delta tables."""

    def __init__(self, spark: SparkSession) -> None:
        self._spark = spark

    def write(self, frame: pl.DataFrame, table: str, mode: WriteMode = "overwrite") -> None:
        spark_df = self._spark.createDataFrame(frame.to_arrow())
        spark_df.write.mode(mode).option("mergeSchema", "true").saveAsTable(table)

    def apply_comments(self, table: str, comments: Mapping[str, str]) -> None:
        try:
            existing = {f.name for f in self._spark.table(table).schema.fields}
        except Exception:
            return
        for column, comment in comments.items():
            if column not in existing:
                continue
            escaped = comment.replace("'", "\\'")
            with contextlib.suppress(Exception):
                self._spark.sql(f"ALTER TABLE {table} ALTER COLUMN `{column}` COMMENT '{escaped}'")


class SparkBinaryFileIndex:
    """``BinaryFileIndex`` backed by Spark's ``binaryFile`` reader + Delta MERGE."""

    def __init__(self, spark: SparkSession) -> None:
        self._spark = spark

    def merge_index(self, directory: Path, table: str) -> int:
        new_df = (
            self._spark.read.format("binaryFile")
            .option("pathGlobFilter", "*.h5")
            .load(str(directory))
            .selectExpr(
                "path",
                # Product code from the filename prefix (RZC, TZC, CPC, BZC, MZC, ...).
                # CombiPrecip reanalysis (CPCH) shares CPC's filename, so it stays "CPC"
                # here — distinguish reanalysis via modification_time, which trails the
                # product time embedded in the filename by ~8 days.
                "regexp_extract(element_at(split(path, '/'), -1), '^[A-Z]+', 0) as product",
                "modificationTime as modification_time",
                "length as size_bytes",
            )
        )

        self._spark.sql(
            f"CREATE TABLE IF NOT EXISTS {table} "
            "(path STRING, product STRING, modification_time TIMESTAMP, size_bytes BIGINT) "
            "USING DELTA"
        )

        view = f"_{directory.name}_new"
        new_df.createOrReplaceTempView(view)
        self._spark.sql(
            f"MERGE INTO {table} t USING {view} s ON t.path = s.path "
            "WHEN MATCHED AND s.modification_time > t.modification_time THEN UPDATE SET "
            "  modification_time = s.modification_time, size_bytes = s.size_bytes "
            "WHEN NOT MATCHED THEN INSERT *"
        )

        return new_df.count()
