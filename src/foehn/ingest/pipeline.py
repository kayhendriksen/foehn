"""Foehn Delta ingestion pipeline.

Public entry points:

- :func:`run_tabular` — ingest CSV/TXT datasets through a :class:`DeltaSink`.
- :func:`run_radar` — index radar HDF5 datasets through a :class:`BinaryFileIndex`.

Both return ``(succeeded, skipped)`` counts. Boot-layer concerns
(``SparkSession``, Unity Catalog DDL) live in ``scripts/ingest_delta.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from foehn.collections import COLLECTIONS, GRIB2_COLLECTIONS, NETCDF_COLLECTIONS
from foehn.convert import _load_metadata_types, parse_csv_bytes
from foehn.ingest._grouping import (
    _build_schema_overrides,
    _group_csv_files,
    _table_suffix,
)
from foehn.ingest.ports import BinaryFileIndex, DeltaSink

TABULAR_COLLECTIONS: tuple[str, ...] = tuple(
    [key for key in COLLECTIONS if key not in GRIB2_COLLECTIONS | NETCDF_COLLECTIONS] + ["climate_normals"]
)

RADAR_COLLECTIONS: tuple[str, ...] = ("radar_precip", "radar_hail")

# Total CSV bytes above which a chunked ingest splits the Arrow→Spark transfer
# (only applies when ``chunked=True``). 2 GiB matches what SMN historical hits.
LARGE_THRESHOLD_BYTES = 2 * 1024**3

DEFAULT_CHUNK_SIZE = 50


def _scan_and_collect(files: list[Path], metadata_types: dict[str, pl.DataType]) -> pl.DataFrame:
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


def _apply_column_comments(sink: DeltaSink, table: str, csv_dir: Path) -> None:
    """Read English column descriptions from ``_meta_parameters.csv`` and forward them to ``sink``."""
    meta_files = list(csv_dir.glob("*_meta_parameters.csv"))
    if not meta_files:
        return

    try:
        meta = pl.read_csv(meta_files[0], separator=";", infer_schema_length=0)
    except Exception:
        return

    required = {"parameter_shortname", "parameter_description_en", "parameter_unit"}
    if not required.issubset(meta.columns):
        return

    comments: dict[str, str] = {}
    for row in meta.select("parameter_shortname", "parameter_description_en", "parameter_unit").iter_rows():
        shortname, desc_en, unit = row
        if not shortname or not desc_en:
            continue
        comments[shortname] = f"{desc_en} [{unit}]" if unit else desc_en

    if comments:
        sink.apply_comments(table, comments)


def _ingest_metadata(
    sink: DeltaSink,
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
            sink.write(df, table)
            print(f"  OK  {tbl_name:25s} → {table} ({df.height} rows)", flush=True)
            succeeded += 1
        except Exception as e:
            print(f"  --  {tbl_name:25s}   skipped ({type(e).__name__})", flush=True)
            skipped += 1

    return succeeded, skipped


def _ingest_collection(
    sink: DeltaSink,
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
    metadata_types = _load_metadata_types(csv_dir)
    groups = _group_csv_files(csv_dir, key)

    meta_ok, meta_skip = _ingest_metadata(sink, key, csv_dir, catalog, schema)

    if not groups:
        print(f"  --  {key:25s}   skipped (no CSVs)", flush=True)
        return meta_ok, meta_skip + 1

    succeeded = meta_ok
    skipped = meta_skip

    for group_key, files in sorted(groups.items()):
        tbl_name = f"{key}{_table_suffix(group_key)}"
        table = f"{catalog}.{schema}.`{tbl_name}`"

        try:
            total_bytes = sum(f.stat().st_size for f in files)
            if chunked and total_bytes > LARGE_THRESHOLD_BYTES and len(files) > chunk_size:
                # Large historical: chunk station files to keep Arrow transfer manageable.
                # Polars streaming handles CSV parsing; chunking bounds the Spark write.
                total_chunks = (len(files) + chunk_size - 1) // chunk_size
                for i in range(0, len(files), chunk_size):
                    chunk = files[i : i + chunk_size]
                    polars_df = _scan_and_collect(chunk, metadata_types)
                    mode = "overwrite" if i == 0 else "append"
                    sink.write(polars_df, table, mode=mode)
                    chunk_num = i // chunk_size + 1
                    print(
                        f"  ... {tbl_name:25s}   chunk {chunk_num}/{total_chunks} ({len(chunk)} files, {mode})",
                        flush=True,
                    )
            else:
                polars_df = _scan_and_collect(files, metadata_types)
                sink.write(polars_df, table)

            print(f"  OK  {tbl_name:25s} → {table} ({len(files)} files)", flush=True)
            succeeded += 1
            _apply_column_comments(sink, table, csv_dir)

        except Exception as e:
            print(f"  --  {tbl_name:25s}   skipped ({type(e).__name__})", flush=True)
            skipped += 1

    return succeeded, skipped


def _ingest_climate_normals(sink: DeltaSink, bronze_base: Path, catalog: str, schema: str) -> tuple[int, int]:
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
    sink.write(combined, table)
    print(f"  OK  {'climate_normals':25s} → {table} ({len(txt_files)} files)", flush=True)
    return 1, 0


def run_tabular(
    bronze_base: Path,
    catalog: str,
    schema: str,
    sink: DeltaSink,
    *,
    keys: Sequence[str] = TABULAR_COLLECTIONS,
    chunked: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[int, int]:
    """Ingest tabular CSV/TXT datasets into Delta tables through ``sink``.

    Iterates ``keys`` (defaults to every tabular dataset). For each:

    - ``climate_normals`` → eager latin1 TXT → one Delta table.
    - any other key → grouped CSVs by (frequency, time slice) plus metadata
      tables; data tables get column comments applied from
      ``_meta_parameters.csv`` after a successful write.

    Datasets whose bronze sub-directory does not exist are counted as
    skipped, never raised. Returns ``(succeeded, skipped)`` totals across
    all keys.
    """
    total_ok = 0
    total_skip = 0

    for key in keys:
        if key == "climate_normals":
            ok, skip = _ingest_climate_normals(sink, bronze_base, catalog, schema)
            total_ok += ok
            total_skip += skip
            continue

        csv_dir = bronze_base / key
        if not csv_dir.exists():
            print(f"  --  {key:25s}   skipped (no data)", flush=True)
            total_skip += 1
            continue

        ok, skip = _ingest_collection(
            sink,
            key,
            csv_dir,
            catalog,
            schema,
            chunked=chunked,
            chunk_size=chunk_size,
        )
        total_ok += ok
        total_skip += skip

    return total_ok, total_skip


def run_radar(
    bronze_base: Path,
    catalog: str,
    schema: str,
    index: BinaryFileIndex,
    *,
    keys: Sequence[str] = RADAR_COLLECTIONS,
) -> tuple[int, int]:
    """Index radar HDF5 datasets into Delta catalog tables through ``index``.

    The HDF5 payload itself stays in place — the table holds
    ``(path, product, modification_time, size_bytes)`` rows that are
    upserted by path. Returns ``(succeeded, skipped)`` totals.
    """
    total_ok = 0
    total_skip = 0

    for key in keys:
        h5_dir = bronze_base / key
        if not h5_dir.exists() or not any(h5_dir.glob("*.h5")):
            print(f"  --  {key:25s} skipped (no data)", flush=True)
            total_skip += 1
            continue

        table = f"{catalog}.{schema}.`{key}`"
        count = index.merge_index(h5_dir, table)
        print(f"  OK  {key:25s} → {table} ({count} files indexed)", flush=True)
        total_ok += 1

    return total_ok, total_skip
