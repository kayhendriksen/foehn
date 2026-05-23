"""Convert downloaded CSVs and TXTs to Parquet using Polars."""

from __future__ import annotations

import io
import re
from pathlib import Path

import polars as pl

_COL_RE = re.compile(r"at column '([^']+)'")

_DTYPE_MAP: dict[str, pl.DataType] = {
    "float": pl.Float64,
    "integer": pl.Int64,
}


def _parse_metadata_types(content: bytes | str) -> dict[str, pl.DataType]:
    """Build a parameter→Polars dtype mapping from metadata CSV content.

    Works with both raw bytes (in-memory) and string content.
    Returns an empty dict if the expected columns are missing.
    """
    try:
        if isinstance(content, str):
            content = content.encode("utf-8")
        meta = pl.read_csv(io.BytesIO(content), separator=";", infer_schema_length=0)
    except Exception:
        return {}

    if "parameter_shortname" not in meta.columns or "parameter_datatype" not in meta.columns:
        return {}

    type_map: dict[str, pl.DataType] = {}
    for row in meta.select("parameter_shortname", "parameter_datatype").iter_rows():
        shortname, datatype = row
        if shortname and datatype:
            dtype = _DTYPE_MAP.get(datatype.strip().lower())
            if dtype is not None:
                type_map[shortname] = dtype
    return type_map


def _load_metadata_types(csv_dir: Path) -> dict[str, pl.DataType]:
    """Build a parameter→Polars dtype mapping from a *_meta_parameters.csv file.

    Returns an empty dict if no metadata file is found or if the expected
    columns (``parameter_shortname``, ``parameter_datatype``) are missing.
    """
    meta_files = list(csv_dir.glob("*_meta_parameters.csv"))
    if not meta_files:
        return {}

    meta_path = meta_files[0]
    try:
        return _parse_metadata_types(meta_path.read_bytes())
    except Exception:
        return {}


def parse_csv_bytes(
    content: bytes,
    metadata_types: dict[str, pl.DataType] | None = None,
    _fallback_overrides: dict[str, pl.DataType] | None = None,
) -> pl.DataFrame:
    """Parse CSV bytes into a Polars DataFrame, applying metadata type overrides.

    Args:
        content: Raw CSV bytes (UTF-8 encoded).
        metadata_types: Optional parameter→dtype mapping from metadata.
        _fallback_overrides: If provided, any Float64 fallback overrides applied
            during error recovery will be written into this dict (for diagnostics).

    Returns:
        Parsed Polars DataFrame.
    """
    buf = io.BytesIO(content)

    # Build per-file overrides by matching CSV columns to metadata types.
    overrides: dict[str, pl.DataType] = {}
    if metadata_types:
        try:
            header = pl.read_csv(io.BytesIO(content), separator=";", n_rows=0, infer_schema_length=0).columns
            for col in header:
                if col in metadata_types:
                    overrides[col] = metadata_types[col]
        except Exception:  # nosec B110
            pass

    try:
        return pl.read_csv(
            buf,
            separator=";",
            infer_schema_length=100,
            try_parse_dates=True,
            schema_overrides=overrides or None,
        )
    except (pl.exceptions.ComputeError, pl.exceptions.SchemaError) as e:
        # Fallback: accumulate Float64 overrides for problematic columns.
        last_err = e
        while True:
            m = _COL_RE.search(str(last_err))
            if not m:
                break
            overrides[m.group(1)] = pl.Float64
            if _fallback_overrides is not None:
                _fallback_overrides[m.group(1)] = pl.Float64
            try:
                return pl.read_csv(
                    io.BytesIO(content),
                    separator=";",
                    infer_schema_length=100,
                    try_parse_dates=True,
                    schema_overrides=overrides,
                )
            except (pl.exceptions.ComputeError, pl.exceptions.SchemaError) as e2:
                last_err = e2
            except Exception:
                raise
        raise last_err from None


def convert_to_parquet(collection_key: str, bronze_dir: Path, parquet_dir: Path) -> int:
    """Convert all CSVs in a collection's bronze folder to combined Parquet files.

    Per-station CSVs are grouped by frequency and time slice, then
    concatenated into a single Parquet file per group.  For example,
    all ``ogd-smn_*_d_recent.csv`` files become ``smn_d_recent.parquet``.

    Args:
        collection_key: Key from COLLECTIONS (e.g. "smn").
        bronze_dir: Root bronze download directory (CSVs in bronze_dir/<key>/).
        parquet_dir: Root parquet directory (output to parquet_dir/<key>/).

    Returns:
        Number of groups that failed to convert. Zero means everything succeeded;
        non-zero lets callers gate downstream state writes (e.g. ``_last_run.json``).
    """
    from foehn.collections import COLLECTIONS, NO_GRANULARITY_COLLECTIONS

    csv_dir = bronze_dir / collection_key
    out_dir = parquet_dir / collection_key
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(csv_dir.glob("*.csv"))
    csv_files = [f for f in csv_files if "_meta_" not in f.name]
    if not csv_files:
        return 0

    # Load parameter type info from metadata once for the whole collection.
    metadata_types = _load_metadata_types(csv_dir)

    # Derive the filename prefix from the collection ID (e.g. "ogd-smn").
    prefix = COLLECTIONS[collection_key].rsplit(".", 1)[-1]
    no_granularity = collection_key in NO_GRANULARITY_COLLECTIONS

    # Group CSVs by (frequency, time_slice).
    groups: dict[tuple[str, ...], list[Path]] = {}
    for csv_path in csv_files:
        suffix_part = csv_path.stem[len(prefix) + 1 :]  # e.g. "ber_d_recent"
        parts = suffix_part.split("_")
        if no_granularity:
            group_key: tuple[str, ...] = ()
        elif len(parts) > 2:
            group_key = (parts[1], parts[2])  # (frequency, time_slice)
        else:
            group_key = (parts[1],) if len(parts) > 1 else ()  # (frequency,)
        groups.setdefault(group_key, []).append(csv_path)

    print(f"Converting {collection_key} to Parquet:", flush=True)
    converted = 0
    skipped = 0
    failed = 0
    for group_key, files in sorted(groups.items()):
        # Output name: smn_d_recent.parquet, smn_d.parquet, or smn.parquet
        if group_key:
            out_name = f"{collection_key}_{'_'.join(group_key)}.parquet"
        else:
            out_name = f"{collection_key}.parquet"
        parquet_path = out_dir / out_name

        # Skip if parquet is already newer than all CSVs in the group.
        if parquet_path.exists():
            parquet_mtime = parquet_path.stat().st_mtime
            if all(f.stat().st_mtime <= parquet_mtime for f in files):
                skipped += 1
                continue

        print(f"  {out_name} ({len(files)} files)...", end="", flush=True)

        # Stay streaming the whole way: on dtype-drift errors (Int inferred from
        # the first N rows but a later row carries a decimal), parse the column
        # name out of the polars error, force it to Float64, and retry.  RSS
        # stays bounded — the alternative of materialising every CSV blew up
        # the driver on historical groups.
        overrides: dict[str, pl.DataType] = dict(metadata_types or {})
        recovered: list[str] = []
        try:
            while True:
                try:
                    lazy_frames = [
                        pl.scan_csv(
                            f,
                            separator=";",
                            try_parse_dates=True,
                            schema_overrides=overrides or None,
                            infer_schema_length=10_000,
                        )
                        for f in files
                    ]
                    pl.concat(lazy_frames, how="diagonal_relaxed").sink_parquet(parquet_path, compression="zstd")
                    break
                except (pl.exceptions.ComputeError, pl.exceptions.SchemaError) as e:
                    m = _COL_RE.search(str(e))
                    if not m:
                        raise
                    col = m.group(1)
                    # If we already forced this column to Float64 and it still
                    # fails, we can't recover by widening dtype further.
                    if overrides.get(col) == pl.Float64:
                        raise
                    overrides[col] = pl.Float64
                    recovered.append(col)
            converted += 1
            if recovered:
                fixed = ", ".join(f"{c}→float" for c in recovered)
                print(f" OK ({fixed})", flush=True)
            else:
                print(" OK", flush=True)
        except Exception as e:
            failed += 1
            print(f" FAIL: {e}", flush=True)

    print(
        f"  Done: {converted} converted, {skipped} skipped (up-to-date), {failed} failed",
        flush=True,
    )
    return failed


_INDOOR_TIME_COLS = ["time.yy", "time.mm", "time.dd", "time.hh"]


def parse_indoor_filename(stem: str) -> tuple[str, str, str, str] | None:
    """Parse an indoor scenario filename into (station, period, scenario, variant).

    Data files are ``{station}_{period}_{scenario}_{variant}`` with a 4-digit
    year as the period. Returns None for anything else (e.g. the archive's
    metadata CSV), so callers can skip non-data files.
    """
    parts = stem.split("_")
    if len(parts) < 4 or not parts[1].isdigit():
        return None
    return parts[0], parts[1], parts[2], "_".join(parts[3:])


def add_indoor_columns(frame, station: str, period: str, scenario: str, variant: str):
    """Add reference_timestamp + filename-derived columns and drop the raw time
    columns. Works on both a LazyFrame (scan_csv) and a DataFrame (read_csv)."""
    return frame.with_columns(
        pl.datetime(
            pl.col("time.yy"),
            pl.col("time.mm"),
            pl.col("time.dd"),
            hour=pl.col("time.hh"),
        ).alias("reference_timestamp"),
        pl.lit(station).alias("station_abbr"),
        pl.lit(period).alias("period"),
        pl.lit(scenario).alias("scenario"),
        pl.lit(variant).alias("variant"),
    ).drop(_INDOOR_TIME_COLS)


def convert_climate_scenarios_indoor_to_parquet(bronze_dir: Path, parquet_dir: Path) -> int:
    """Convert extracted indoor climate scenario CSVs to a single Parquet file.

    Each CSV is per-station/per-scenario hourly data: comma-separated, with the
    timestamp split across time.yy/mm/dd/hh. The filename encodes
    {station}_{period}_{scenario}_{variant}, which become columns alongside a
    synthesised reference_timestamp. All files are concatenated into one Parquet.
    """
    csv_dir = bronze_dir / "climate_scenarios_indoor"
    out_dir = parquet_dir / "climate_scenarios_indoor"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        return 0

    out_path = out_dir / "climate_scenarios_indoor.parquet"
    if out_path.exists():
        out_mtime = out_path.stat().st_mtime
        if all(f.stat().st_mtime <= out_mtime for f in csv_files):
            print("Converting climate_scenarios_indoor to Parquet:\n  up-to-date — skipping", flush=True)
            return 0

    print(f"Converting climate_scenarios_indoor to Parquet ({len(csv_files)} files)...", flush=True)
    frames: list[pl.LazyFrame] = []
    skipped = 0
    for f in csv_files:
        parsed = parse_indoor_filename(f.stem)
        if parsed is None:
            skipped += 1
            print(f"  Skipping non-data file: {f.name}", flush=True)
            continue
        station, period, scenario, variant = parsed
        lf = add_indoor_columns(
            pl.scan_csv(f, separator=",", infer_schema_length=10_000, truncate_ragged_lines=True),
            station,
            period,
            scenario,
            variant,
        )
        frames.append(lf)

    if not frames:
        return 0

    try:
        pl.concat(frames, how="diagonal_relaxed").sink_parquet(out_path, compression="zstd")
    except Exception as e:
        print(f"  FAIL: {e}", flush=True)
        return 1

    print(f"  Done: wrote {out_path.name} ({len(frames)} files, {skipped} non-data skipped)", flush=True)
    return 0


def convert_climate_normals_to_parquet(bronze_dir: Path, parquet_dir: Path) -> int:
    """Convert C6 climate normals TXT files to Parquet.

    These files use tab separators, latin1 encoding, and have 7 header rows
    to skip before the actual data begins.
    """
    txt_dir = bronze_dir / "climate_normals"
    out_dir = parquet_dir / "climate_normals"
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(txt_dir.glob("*.txt"))
    if not txt_files:
        return 0

    print("Converting climate_normals to Parquet:", flush=True)
    converted = 0
    skipped = 0
    failed = 0
    total = len(txt_files)
    for i, txt_path in enumerate(txt_files, 1):
        parquet_path = out_dir / txt_path.with_suffix(".parquet").name

        if parquet_path.exists() and parquet_path.stat().st_mtime >= txt_path.stat().st_mtime:
            skipped += 1
            continue

        print(f"  [{i}/{total}] {txt_path.name}...", end="", flush=True)
        try:
            df = pl.read_csv(
                txt_path,
                separator="\t",
                skip_rows=8,
                encoding="latin1",
                infer_schema_length=None,
                try_parse_dates=True,
                truncate_ragged_lines=True,
            )
            df.write_parquet(parquet_path, compression="snappy")
            converted += 1
            print(" Converted", flush=True)
        except Exception as e:
            failed += 1
            print(f" FAIL: {e}", flush=True)

    print(
        f"  Done: {converted} converted, {skipped} skipped (up-to-date), {failed} failed",
        flush=True,
    )
    return failed
