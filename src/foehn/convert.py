"""Bronze → Parquet: one output per group, written atomically, skipped when current.

The convert stage of the pipeline, and nothing else. How a MeteoSwiss CSV is
decoded, typed and grouped is :mod:`foehn.meteocsv`, which this module imports
and the load path imports separately — the two stages share upstream conventions
without either owning the other.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl

from foehn.meteocsv import (
    add_forecast_local_timestamp,
    add_indoor_columns,
    column_from_dtype_error,
    group_csv_files,
    load_metadata_types,
    parse_indoor_filename,
    scan_climate_scenarios_csv,
)
from foehn.workspace import Workspace

logger = logging.getLogger(__name__)

# The two codecs the converters below actually use. Narrower than polars'
# ParquetCompression on purpose: a bare ``str`` is wider than the literal
# polars accepts, which the type checker rejects at the call site.
_ParquetCompression = Literal["zstd", "snappy"]


def _write_parquet_atomic(
    frame: pl.LazyFrame | pl.DataFrame, path: Path, compression: _ParquetCompression = "zstd"
) -> None:
    """Write a frame to *path* via a sibling temp file + Path.replace.

    A write that dies part-way (disk full, a source read error, OOM mid-stream)
    creates the output file before it fails. Writing straight to the final path
    would leave that truncated Parquet behind *with a fresh mtime* — which the
    up-to-date checks in the converters below read as "already converted" and
    skip from then on, so the next run reports success over a corrupt file.
    Staging into a temp file means a failed write leaves nothing at all.
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        if isinstance(frame, pl.LazyFrame):
            frame.sink_parquet(tmp, compression=compression)
        else:
            frame.write_parquet(tmp, compression=compression)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class ConversionGroup:
    """One Parquet output and the source files it is built from."""

    out_path: Path
    sources: list[Path]


ConvertOne = Callable[[ConversionGroup], int]


def _is_up_to_date(group: ConversionGroup) -> bool:
    """True when the output exists and is at least as new as every source."""
    if not group.out_path.exists():
        return False
    out_mtime = group.out_path.stat().st_mtime
    return all(f.stat().st_mtime <= out_mtime for f in group.sources)


def run_conversions(groups: Iterable[ConversionGroup], convert_one: ConvertOne, *, label: str) -> int:
    """Run each group's conversion, skipping outputs that are already current.

    The four converters below each used to carry their own copy of this: the
    mtime up-to-date check, the converted/skipped/failed counters, the per-group
    try/except and the summary line. What actually differs between them is how a
    frame is built — which is ``convert_one``, and stays with each kind.

    Returns:
        Total failures: groups that raised, plus the source-level failures the
        conversions themselves reported. Zero means everything succeeded, which
        is what gates ``_last_run.json``.
    """
    groups = list(groups)
    if not groups:
        return 0

    logger.info("Converting %s to Parquet:", label)
    converted = skipped = failed = 0
    for group in sorted(groups, key=lambda g: g.out_path.name):
        if _is_up_to_date(group):
            skipped += 1
            continue
        group.out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            failed += convert_one(group)
        except Exception as e:
            failed += 1
            logger.warning("  %s (%d files)... FAIL: %s", group.out_path.name, len(group.sources), e)
            continue
        if group.out_path.exists():
            converted += 1

    logger.info("  Done: %d converted, %d skipped (up-to-date), %d failed", converted, skipped, failed)
    return failed


def _group_out_name(collection_key: str, group_key: tuple[str, ...]) -> str:
    """Output name for one group: smn_d_recent.parquet, smn_d.parquet, or smn.parquet."""
    return f"{collection_key}_{'_'.join(group_key)}.parquet" if group_key else f"{collection_key}.parquet"


def _convert_standard_group(collection_key: str, metadata_types: dict[str, type[pl.DataType]]) -> ConvertOne:
    """Build the per-group converter for standard CSV collections.

    Stays streaming the whole way: on dtype-drift errors (Int inferred from the
    first N rows but a later row carries a decimal), parse the column name out of
    the polars error, force it to Float64, and retry. RSS stays bounded — the
    alternative of materialising every CSV blew up the driver on historical
    groups. The retry has to wrap the write, not just the scan: ``scan_csv`` is
    lazy, so the schema error only surfaces when ``sink_parquet`` pulls the rows.
    """

    def convert_one(group: ConversionGroup) -> int:
        overrides: dict[str, type[pl.DataType]] = dict(metadata_types or {})
        recovered: list[str] = []
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
                    for f in group.sources
                ]
                combined = pl.concat(lazy_frames, how="diagonal_relaxed")
                if collection_key == "forecast_local":
                    combined = add_forecast_local_timestamp(combined)
                _write_parquet_atomic(combined, group.out_path)
                break
            except (pl.exceptions.ComputeError, pl.exceptions.SchemaError) as e:
                col = column_from_dtype_error(str(e))
                if col is None:
                    raise
                # If we already forced this column to Float64 and it still
                # fails, we can't recover by widening dtype further.
                if overrides.get(col) == pl.Float64:
                    raise
                overrides[col] = pl.Float64
                recovered.append(col)
        if recovered:
            fixed = ", ".join(f"{c}→float" for c in recovered)
            logger.info("  %s (%d files)... OK (%s)", group.out_path.name, len(group.sources), fixed)
        else:
            logger.info("  %s (%d files)... OK", group.out_path.name, len(group.sources))
        return 0

    return convert_one


def convert_to_parquet(dataset: str, workspace: Workspace) -> int:
    """Convert all CSVs in a collection's bronze folder to combined Parquet files.

    Per-station CSVs are grouped by frequency and time slice, then
    concatenated into a single Parquet file per group.  For example,
    all ``ogd-smn_*_d_recent.csv`` files become ``smn_d_recent.parquet``.

    Args:
        dataset: Key from COLLECTIONS (e.g. "smn").
        workspace: Where the CSVs are read from and the Parquet is written.

    Returns:
        Number of groups that failed to convert. Zero means everything succeeded;
        non-zero lets callers gate downstream state writes (e.g. ``_last_run.json``).
    """
    csv_dir = workspace.bronze(dataset)
    out_dir = workspace.parquet(dataset)

    groups = [
        ConversionGroup(out_dir / _group_out_name(dataset, group_key), files)
        for group_key, files in group_csv_files(csv_dir, dataset).items()
    ]
    if not groups:
        return 0

    # Load parameter type info from metadata once for the whole collection.
    metadata_types = load_metadata_types(csv_dir)
    return run_conversions(groups, _convert_standard_group(dataset, metadata_types), label=dataset)


def convert_indoor_to_parquet(dataset: str, workspace: Workspace) -> int:
    """Convert extracted indoor climate scenario CSVs to a single Parquet file.

    Each CSV is per-station/per-scenario hourly data: comma-separated, with the
    timestamp split across time.yy/mm/dd/hh. The filename encodes
    {station}_{period}_{scenario}_{variant}, which become columns alongside a
    synthesised reference_timestamp. All files are concatenated into one Parquet.
    """
    csv_files = sorted(workspace.bronze(dataset).glob("*.csv"))
    if not csv_files:
        return 0

    def convert_one(group: ConversionGroup) -> int:
        frames: list[pl.LazyFrame] = []
        skipped = 0
        for f in group.sources:
            parsed = parse_indoor_filename(f.stem)
            if parsed is None:
                # The archive ships a metadata CSV alongside the data; not a failure.
                skipped += 1
                logger.info("  Skipping non-data file: %s", f.name)
                continue
            station, period, scenario, variant = parsed
            frames.append(
                add_indoor_columns(
                    pl.scan_csv(f, separator=",", infer_schema_length=10_000, truncate_ragged_lines=True),
                    station,
                    period,
                    scenario,
                    variant,
                )
            )
        if not frames:
            return 0
        _write_parquet_atomic(pl.concat(frames, how="diagonal_relaxed"), group.out_path)
        logger.info("  Done: wrote %s (%d files, %d non-data skipped)", group.out_path.name, len(frames), skipped)
        return 0

    out_path = workspace.parquet(dataset) / f"{dataset}.parquet"
    return run_conversions([ConversionGroup(out_path, csv_files)], convert_one, label=dataset)


def convert_preamble_to_parquet(dataset: str, workspace: Workspace) -> int:
    """Convert CH2025 climate-scenario CSVs (with metadata preamble) to one Parquet."""
    csv_files = sorted(f for f in workspace.bronze(dataset).glob("*.csv") if "_meta_" not in f.name)
    if not csv_files:
        return 0

    def convert_one(group: ConversionGroup) -> int:
        frames: list[pl.LazyFrame] = []
        failed = 0
        for f in group.sources:
            try:
                frames.append(scan_climate_scenarios_csv(f))
            except Exception as e:
                # One unreadable file must not cost the rest of the collection —
                # skip it, write what parsed, and report it so the exit code
                # still reflects it.
                failed += 1
                logger.warning("  FAIL %s: %s", f.name, e)
        if not frames:
            return failed
        _write_parquet_atomic(pl.concat(frames, how="diagonal_relaxed"), group.out_path)
        logger.info("  Done: wrote %s (%d files)", group.out_path.name, len(frames))
        return failed

    out_path = workspace.parquet(dataset) / f"{dataset}.parquet"
    return run_conversions([ConversionGroup(out_path, csv_files)], convert_one, label=dataset)


def convert_normals_to_parquet(dataset: str, workspace: Workspace) -> int:
    """Convert C6 climate normals TXT files to Parquet.

    These files use tab separators, latin1 encoding, and have 8 header rows
    to skip before the actual data begins.
    """
    txt_files = sorted(workspace.bronze(dataset).glob("*.txt"))
    if not txt_files:
        return 0

    def convert_one(group: ConversionGroup) -> int:
        source = group.sources[0]
        df = pl.read_csv(
            source,
            separator="\t",
            skip_rows=8,
            encoding="latin1",
            infer_schema_length=None,
            try_parse_dates=True,
            truncate_ragged_lines=True,
        )
        _write_parquet_atomic(df, group.out_path, compression="snappy")
        logger.info("  %s... Converted", source.name)
        return 0

    out_dir = workspace.parquet(dataset)
    groups = [ConversionGroup(out_dir / txt.with_suffix(".parquet").name, [txt]) for txt in txt_files]
    return run_conversions(groups, convert_one, label=dataset)
