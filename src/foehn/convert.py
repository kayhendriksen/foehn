"""Convert downloaded CSVs and TXTs to Parquet using Polars."""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

_COL_RE = re.compile(r"at column '([^']+)'")


def decode_meteoswiss_csv(content: bytes) -> str:
    """Decode MeteoSwiss CSV bytes to text.

    MeteoSwiss CSVs are usually UTF-8 (often with a BOM) but some legacy files
    are Windows-1252. Try UTF-8 (BOM-aware) first, falling back to Windows-1252.
    """
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return content.decode("windows-1252")


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


def add_forecast_local_timestamp(frame):
    """Add a parsed reference_timestamp from forecast_local's compact Date column.

    forecast_local CSVs carry ``point_id;point_type_id;Date;<param>`` where Date
    is ``YYYYMMDDHHMM`` (e.g. 202605202100) and there is no reference_timestamp.
    Works on both a LazyFrame and a DataFrame; a no-op if Date is absent or a
    reference_timestamp already exists.
    """
    cols = frame.collect_schema().names() if isinstance(frame, pl.LazyFrame) else frame.columns
    if "Date" not in cols or "reference_timestamp" in cols:
        return frame
    return frame.with_columns(
        pl.col("Date").cast(pl.Utf8).str.to_datetime("%Y%m%d%H%M", strict=False).alias("reference_timestamp")
    )


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

    logger.info("Converting %s to Parquet:", collection_key)
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
                    combined = pl.concat(lazy_frames, how="diagonal_relaxed")
                    if collection_key == "forecast_local":
                        combined = add_forecast_local_timestamp(combined)
                    combined.sink_parquet(parquet_path, compression="zstd")
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
                logger.info("  %s (%d files)... OK (%s)", out_name, len(files), fixed)
            else:
                logger.info("  %s (%d files)... OK", out_name, len(files))
        except Exception as e:
            failed += 1
            logger.warning("  %s (%d files)... FAIL: %s", out_name, len(files), e)

    logger.info("  Done: %d converted, %d skipped (up-to-date), %d failed", converted, skipped, failed)
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
            logger.info("Converting climate_scenarios_indoor to Parquet: up-to-date — skipping")
            return 0

    logger.info("Converting climate_scenarios_indoor to Parquet (%d files)...", len(csv_files))
    frames: list[pl.LazyFrame] = []
    skipped = 0
    for f in csv_files:
        parsed = parse_indoor_filename(f.stem)
        if parsed is None:
            skipped += 1
            logger.info("  Skipping non-data file: %s", f.name)
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
        logger.warning("  FAIL: %s", e)
        return 1

    logger.info("  Done: wrote %s (%d files, %d non-data skipped)", out_path.name, len(frames), skipped)
    return 0


def parse_climate_scenarios_csv(content: bytes | str, filename: str) -> pl.DataFrame:
    """Parse a CH2025 climate-scenario CSV into a wide model table.

    These files carry a multi-row ``KEY;VALUE`` metadata preamble before the
    real ``DATE;<model>;<model>;...`` header. We skip the preamble, read the
    table, and tag rows with station/variable/gwl parsed from the filename
    (``ogd-climate-scenarios-ch2025_{station}_{variable}_{gwl}``). The DATE
    column is kept as a string because it encodes a nominal 30-year period
    (0001-01-01 … 0030-12-31 on a 365-day calendar), not real calendar dates.
    """
    text = content.decode("utf-8-sig", errors="replace") if isinstance(content, bytes) else content
    lines = text.splitlines()
    header_idx = next((i for i, line in enumerate(lines) if line.startswith("DATE;")), None)
    if header_idx is None:
        raise ValueError(f"No 'DATE;' data header found in {filename!r}")

    table = "\n".join(lines[header_idx:])
    df = pl.read_csv(
        io.BytesIO(table.encode("utf-8")),
        separator=";",
        infer_schema_length=20_000,
        truncate_ragged_lines=True,
    ).rename({"DATE": "date"})

    stem = filename.rsplit("/", 1)[-1]
    stem = stem[:-4] if stem.endswith(".csv") else stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(
            f"Unexpected climate-scenario filename {filename!r}: expected "
            "'..._{station}_{variable}_{gwl}', cannot extract station/variable/gwl."
        )
    station, variable, gwl = parts[-3], parts[-2], parts[-1]

    df = df.with_columns(
        pl.lit(station).alias("station_abbr"),
        pl.lit(variable).alias("variable"),
        pl.lit(gwl).alias("gwl"),
    )
    front = ["station_abbr", "variable", "gwl", "date"]
    return df.select(front + [c for c in df.columns if c not in front])


def convert_climate_scenarios_to_parquet(bronze_dir: Path, parquet_dir: Path) -> int:
    """Convert CH2025 climate-scenario CSVs (with metadata preamble) to one Parquet."""
    csv_dir = bronze_dir / "climate_scenarios"
    out_dir = parquet_dir / "climate_scenarios"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(f for f in csv_dir.glob("*.csv") if "_meta_" not in f.name)
    if not csv_files:
        return 0

    out_path = out_dir / "climate_scenarios.parquet"
    if out_path.exists():
        out_mtime = out_path.stat().st_mtime
        if all(f.stat().st_mtime <= out_mtime for f in csv_files):
            logger.info("Converting climate_scenarios to Parquet: up-to-date — skipping")
            return 0

    def _safe_parse(path: Path) -> pl.DataFrame | None:
        try:
            return parse_climate_scenarios_csv(path.read_bytes(), path.name)
        except Exception as e:
            logger.warning("  FAIL %s: %s", path.name, e)
            return None

    logger.info("Converting climate_scenarios to Parquet (%d files)...", len(csv_files))
    parsed = [_safe_parse(f) for f in csv_files]
    frames = [df for df in parsed if df is not None]
    failed = sum(1 for df in parsed if df is None)

    if not frames:
        return failed

    try:
        pl.concat(frames, how="diagonal_relaxed").write_parquet(out_path, compression="zstd")
    except Exception as e:
        logger.warning("  FAIL: %s", e)
        return failed + 1

    logger.info("  Done: wrote %s (%d files)", out_path.name, len(frames))
    return failed


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

    logger.info("Converting climate_normals to Parquet:")
    converted = 0
    skipped = 0
    failed = 0
    total = len(txt_files)
    for i, txt_path in enumerate(txt_files, 1):
        parquet_path = out_dir / txt_path.with_suffix(".parquet").name

        if parquet_path.exists() and parquet_path.stat().st_mtime >= txt_path.stat().st_mtime:
            skipped += 1
            continue

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
            logger.info("  [%d/%d] %s... Converted", i, total, txt_path.name)
        except Exception as e:
            failed += 1
            logger.warning("  [%d/%d] %s... FAIL: %s", i, total, txt_path.name, e)

    logger.info("  Done: %d converted, %d skipped (up-to-date), %d failed", converted, skipped, failed)
    return failed
