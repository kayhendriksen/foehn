"""Ingest MeteoSwiss CSVs directly into Delta tables via Polars + Spark.

Reads bronze CSV files from a Unity Catalog Volume, lazily scans them with
Polars (scan_csv → LazyFrame), concatenates per group, and collects with
engine="streaming" to keep peak memory low — even for collections that
exceed RAM (e.g. SMN historical ~20 GB of 10-minute data).

The resulting DataFrame is converted to a Spark DataFrame via Arrow
(zero-copy) and written to a Delta table.  For large collections in
historical mode, station files are chunked so that each Arrow→Spark
transfer stays manageable.

After writing each table, column comments are set from the English
metadata descriptions in _meta_parameters.csv.

Run by the Databricks job after foehn finishes downloading.

Usage (local testing with spark-submit):
    spark-submit scripts/ingest_delta.py --catalog main --schema meteoswiss --volume landing
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
from pathlib import Path

import polars as pl
from pyspark.sql import SparkSession

from foehn import registry
from foehn.convert import group_csv_files, load_metadata_types, parse_csv_bytes

TABULAR_COLLECTIONS = [*registry.tabular_datasets(), "climate_normals"]

# Collections where historical data can exceed available memory.
# Chunked ingestion is used for these when --historical is set.
LARGE_COLLECTIONS = {"smn", "smn_precip", "smn_tower"}

DEFAULT_CHUNK_SIZE = 50

_IDENTIFIER_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")


def _validate_identifier(value: str, label: str) -> str:
    if not _IDENTIFIER_RE.match(value):
        raise ValueError(f"Invalid {label} {value!r} — only alphanumerics, underscores, and hyphens are allowed")
    return f"`{value}`"


def _table_suffix(group_key: tuple[str, ...]) -> str:
    if group_key:
        return f"_{'_'.join(group_key)}"
    return ""


def _build_schema_overrides(
    files: list[Path], metadata_types: dict[str, type[pl.DataType]]
) -> dict[str, type[pl.DataType]] | None:
    """Match CSV column headers to metadata types for schema overrides."""
    if not metadata_types:
        return None
    try:
        header = pl.read_csv(files[0], separator=";", n_rows=0, infer_schema_length=0).columns
        overrides = {col: metadata_types[col] for col in header if col in metadata_types}
        return overrides or None
    except Exception:
        return None


def _scan_and_collect(files: list[Path], metadata_types: dict[str, type[pl.DataType]]) -> pl.DataFrame:
    """Lazily scan CSV files and collect with streaming engine.

    Falls back to eager parse_csv_bytes (with retry logic) if streaming
    collect fails — e.g. when a column has mixed types that only the
    fallback's per-column Float64 override can handle.
    """
    overrides = _build_schema_overrides(files, metadata_types)

    lazy_frames: list[pl.LazyFrame] = []
    for f in files:
        lf = pl.scan_csv(
            f,
            separator=";",
            infer_schema_length=100,
            try_parse_dates=True,
            schema_overrides=overrides,
        )
        lazy_frames.append(lf)

    combined = pl.concat(lazy_frames, how="diagonal_relaxed")
    try:
        return combined.collect(engine="streaming")
    except (pl.exceptions.ComputeError, pl.exceptions.SchemaError):
        # Fall back to eager parsing with parse_csv_bytes retry logic.
        frames = [parse_csv_bytes(f.read_bytes(), metadata_types) for f in files]
        return pl.concat(frames, how="diagonal_relaxed")


def _write_to_delta(spark: SparkSession, polars_df: pl.DataFrame, table: str, mode: str = "overwrite") -> None:
    """Convert a Polars DataFrame to Spark via Arrow and write to a Delta table."""
    spark_df = spark.createDataFrame(polars_df.to_arrow())
    spark_df.write.mode(mode).option("mergeSchema", "true").saveAsTable(table)


def _apply_column_comments(spark: SparkSession, table: str, csv_dir: Path) -> None:
    """Set Delta column comments from the English metadata descriptions.

    Reads _meta_parameters.csv and applies comments like:
        "Daily mean air temperature 2 m above ground [°C]"
    """
    meta_files = sorted(csv_dir.glob("*_meta_parameters.csv"))
    if not meta_files:
        return

    try:
        meta = pl.read_csv(meta_files[0], separator=";", infer_schema_length=0)
    except Exception:
        return

    required = {"parameter_shortname", "parameter_description_en", "parameter_unit"}
    if not required.issubset(meta.columns):
        return

    try:
        table_cols = {f.name for f in spark.table(table).schema.fields}
    except Exception:
        return

    for row in meta.select("parameter_shortname", "parameter_description_en", "parameter_unit").iter_rows():
        shortname, desc_en, unit = row
        if not shortname or not desc_en or shortname not in table_cols:
            continue
        comment = f"{desc_en} [{unit}]" if unit else desc_en
        # Escape backslashes before quotes — a trailing ``\'`` in the metadata
        # would otherwise escape the escape and close the SQL string literal.
        comment_escaped = comment.replace("\\", "\\\\").replace("'", "\\'")
        with contextlib.suppress(Exception):
            spark.sql(f"ALTER TABLE {table} ALTER COLUMN `{shortname}` COMMENT '{comment_escaped}'")


def _ingest_metadata(
    spark: SparkSession,
    key: str,
    csv_dir: Path,
    catalog: str,
    schema: str,
) -> tuple[int, int]:
    """Ingest *_meta_*.csv files (stations, parameters, datainventory, ...) into Delta.

    Each meta CSV becomes its own table: ``{collection}_meta_{suffix}``. Always
    overwrites — these files are small, authoritative, and refreshed each run.
    """
    meta_files = sorted(csv_dir.glob("*_meta_*.csv"))
    succeeded = 0
    skipped = 0

    for meta_path in meta_files:
        # "ogd-smn_meta_stations.csv" → suffix "stations"
        suffix = meta_path.stem.split("_meta_", 1)[1]
        tbl_name = f"{key}_meta_{suffix}"
        table = f"{catalog}.{schema}.`{tbl_name}`"

        try:
            df = pl.read_csv(
                meta_path,
                separator=";",
                infer_schema_length=10000,
                try_parse_dates=True,
            )
            _write_to_delta(spark, df, table)
            print(f"  OK  {tbl_name:25s} → {table} ({df.height} rows)", flush=True)
            succeeded += 1
        except Exception as e:
            print(f"  --  {tbl_name:25s}   skipped ({type(e).__name__})", flush=True)
            skipped += 1

    return succeeded, skipped


def _ingest_collection(
    spark: SparkSession,
    key: str,
    csv_dir: Path,
    catalog: str,
    schema: str,
    *,
    chunked: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[int, int]:
    """Ingest a single collection's CSVs into Delta tables.

    Returns (succeeded, skipped) counts.
    """
    metadata_types = load_metadata_types(csv_dir)
    groups = group_csv_files(csv_dir, key)

    meta_ok, meta_skip = _ingest_metadata(spark, key, csv_dir, catalog, schema)

    if not groups:
        print(f"  --  {key:25s}   skipped (no CSVs)", flush=True)
        return meta_ok, meta_skip + 1

    succeeded = meta_ok
    skipped = meta_skip

    for group_key, files in sorted(groups.items()):
        tbl_name = f"{key}{_table_suffix(group_key)}"
        table = f"{catalog}.{schema}.`{tbl_name}`"

        try:
            if chunked and key in LARGE_COLLECTIONS and len(files) > chunk_size:
                # Large historical: chunk station files to keep Arrow transfer manageable.
                # Polars streaming handles CSV parsing; chunking bounds the Spark write.
                total_chunks = (len(files) + chunk_size - 1) // chunk_size
                for i in range(0, len(files), chunk_size):
                    chunk = files[i : i + chunk_size]
                    polars_df = _scan_and_collect(chunk, metadata_types)
                    mode = "overwrite" if i == 0 else "append"
                    _write_to_delta(spark, polars_df, table, mode=mode)
                    chunk_num = i // chunk_size + 1
                    print(
                        f"  ... {tbl_name:25s}   chunk {chunk_num}/{total_chunks} ({len(chunk)} files, {mode})",
                        flush=True,
                    )
            else:
                polars_df = _scan_and_collect(files, metadata_types)
                _write_to_delta(spark, polars_df, table)

            print(f"  OK  {tbl_name:25s} → {table} ({len(files)} files)", flush=True)
            succeeded += 1
            _apply_column_comments(spark, table, csv_dir)

        except Exception as e:
            print(f"  --  {tbl_name:25s}   skipped ({type(e).__name__})", flush=True)
            skipped += 1

    return succeeded, skipped


RADAR_COLLECTIONS = ("radar_precip", "radar_hail")


def _ingest_radar(
    spark: SparkSession,
    bronze_base: Path,
    cat: str,
    sch: str,
    keys: tuple[str, ...] = RADAR_COLLECTIONS,
) -> tuple[int, int]:
    """Index radar HDF5 files into per-collection Delta catalog tables.

    Scans each collection's landing dir via Spark's binaryFile reader and
    MERGEs by path, so repeated 5-min runs only insert newly-arrived files.
    The HDF5 payload itself stays in the volume — the table just holds
    (path, modification_time, size_bytes).
    """
    total_ok = 0
    total_skip = 0

    for key in keys:
        h5_dir = bronze_base / key
        if not h5_dir.exists() or not any(h5_dir.glob("*.h5")):
            print(f"  --  {key:25s} skipped (no data)", flush=True)
            total_skip += 1
            continue

        new_df = (
            spark.read.format("binaryFile")
            .option("pathGlobFilter", "*.h5")
            .load(str(h5_dir))
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

        table = f"{cat}.{sch}.`{key}`"
        spark.sql(
            f"CREATE TABLE IF NOT EXISTS {table} "
            "(path STRING, product STRING, modification_time TIMESTAMP, size_bytes BIGINT) "
            "USING DELTA"
        )

        view = f"_{key}_new"
        new_df.createOrReplaceTempView(view)
        # WHEN MATCHED: pick up reanalysis overwrites (local mtime bumped, row should too).
        # The interpolated names are not user-controlled SQL: `table` is built from
        # `cat`/`sch`, which both passed _validate_identifier (alphanumerics, underscore,
        # hyphen; backtick-quoted), and from `key`, which the caller constrains to
        # RADAR_COLLECTIONS literals. `view` derives from that same literal. Hence the
        # suppression on the MERGE below.
        merge_sql = (
            f"MERGE INTO {table} t USING {view} s ON t.path = s.path "  # nosec B608
            "WHEN MATCHED AND s.modification_time > t.modification_time THEN UPDATE SET "
            "  modification_time = s.modification_time, size_bytes = s.size_bytes "
            "WHEN NOT MATCHED THEN INSERT *"
        )
        spark.sql(merge_sql)

        count = new_df.count()
        print(f"  OK  {key:25s} → {table} ({count} files indexed)", flush=True)
        total_ok += 1

    return total_ok, total_skip


def _ingest_climate_normals(spark: SparkSession, bronze_base: Path, catalog: str, schema: str) -> tuple[int, int]:
    """Ingest climate normals TXT files into a single Delta table."""
    txt_dir = bronze_base / "climate_normals"
    if not txt_dir.exists():
        print("  --  climate_normals           skipped (no data)", flush=True)
        return 0, 1

    txt_files = sorted(txt_dir.glob("*.txt"))
    if not txt_files:
        print("  --  climate_normals           skipped (no TXT files)", flush=True)
        return 0, 1

    # scan_csv doesn't support encoding="latin1", so use eager read_csv.
    frames: list[pl.DataFrame] = []
    for txt_path in txt_files:
        try:
            df = pl.read_csv(
                txt_path,
                separator="\t",
                skip_rows=8,
                encoding="latin1",
                infer_schema_length=100,
                try_parse_dates=True,
                truncate_ragged_lines=True,
            )
            frames.append(df)
        except Exception as e:
            print(f"  WARN: {txt_path.name}: {e}", flush=True)

    if not frames:
        return 0, 1

    table = f"{catalog}.{schema}.`climate_normals`"
    combined = pl.concat(frames, how="diagonal_relaxed")
    _write_to_delta(spark, combined, table)
    print(f"  OK  {'climate_normals':25s} → {table} ({len(txt_files)} files)", flush=True)
    return 1, 0


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

    bronze_base = Path(f"/Volumes/{args.catalog}/{args.schema}/{args.volume}/bronze")

    total_ok = 0
    total_skip = 0

    if args.radar is not None:
        keys = tuple(args.radar) if args.radar else RADAR_COLLECTIONS
        invalid = [k for k in keys if k not in RADAR_COLLECTIONS]
        if invalid:
            raise SystemExit(f"Unknown radar collection(s): {invalid}. Valid: {list(RADAR_COLLECTIONS)}")
        ok, skip = _ingest_radar(spark, bronze_base, cat, sch, keys)
        print(f"\nDone — {ok} tables written, {skip} skipped.")
        return

    for key in TABULAR_COLLECTIONS:
        if key == "climate_normals":
            ok, skip = _ingest_climate_normals(spark, bronze_base, cat, sch)
            total_ok += ok
            total_skip += skip
            continue

        csv_dir = bronze_base / key
        if not csv_dir.exists():
            print(f"  --  {key:25s}   skipped (no data)", flush=True)
            total_skip += 1
            continue

        ok, skip = _ingest_collection(
            spark,
            key,
            csv_dir,
            cat,
            sch,
            chunked=args.historical,
            chunk_size=args.chunk_size,
        )
        total_ok += ok
        total_skip += skip

    print(f"\nDone — {total_ok} tables written, {total_skip} skipped.")


if __name__ == "__main__":
    main()
