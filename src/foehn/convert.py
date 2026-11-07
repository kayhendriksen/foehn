"""Bronze → Parquet: one output per group, written atomically, skipped when current.

The convert stage of the pipeline, and nothing else. How a MeteoSwiss CSV is
decoded, typed and grouped is :mod:`foehn.meteocsv`, which this module imports
and the load path imports separately — the two stages share upstream conventions
without either owning the other.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl

from foehn.atomicwrite import staged
from foehn.meteocsv import (
    column_from_dtype_error,
    combine_frames,
    derive_timestamp,
    group_csv_files,
    indoor_station,
    load_metadata_types,
    read_normals_txt,
    scan_climate_scenarios_csv,
    scan_indoor_csv,
    scan_standard_csv,
)
from foehn.retention import retired_groups
from foehn.workspace import Workspace

logger = logging.getLogger(__name__)

# The two codecs the converters below actually use. Narrower than polars'
# ParquetCompression on purpose: a bare ``str`` is wider than the literal
# polars accepts, which the type checker rejects at the call site.
_ParquetCompression = Literal["zstd", "snappy"]


# The footer key naming the Bronze files an output was built from.
_SOURCES_KEY = "foehn_sources"


def _source_names(sources: Iterable[Path]) -> list[str]:
    return sorted(path.name for path in sources)


def _write_parquet_atomic(
    frame: pl.LazyFrame | pl.DataFrame,
    path: Path,
    compression: _ParquetCompression = "zstd",
    *,
    sources: list[Path],
) -> None:
    """Write a frame to *path* via a sibling temp file + Path.replace.

    A write that dies part-way (disk full, a source read error, OOM mid-stream)
    creates the output file before it fails. Writing straight to the final path
    would leave that truncated Parquet behind *with a fresh mtime* — which the
    up-to-date checks in the converters below read as "already converted" and
    skip from then on, so the next run reports success over a corrupt file.
    Staging into a temp file means a failed write leaves nothing at all.

    The footer records which *sources* the output was built from, so a source
    deleted from Bronze is noticed: every one left behind is older than the
    output, and by mtime alone the stale table was current.
    """
    metadata = {_SOURCES_KEY: json.dumps(_source_names(sources))}
    with staged(path) as tmp:
        if isinstance(frame, pl.LazyFrame):
            frame.sink_parquet(tmp, compression=compression, metadata=metadata)
        else:
            frame.write_parquet(tmp, compression=compression, metadata=metadata)


@dataclass(frozen=True)
class ConversionGroup:
    """One Parquet output and the source files it is built from."""

    out_path: Path
    sources: list[Path]


ConvertOne = Callable[[ConversionGroup], int]


def _built_from(path: Path, sources: list[Path]) -> bool:
    """Whether the output at *path* was built from exactly *sources*."""
    try:
        recorded = pl.read_parquet_metadata(path).get(_SOURCES_KEY)
    except Exception:  # noqa: BLE001 - an unreadable footer is a reason to rebuild, never to fail
        return False
    if recorded is None:
        # Written before outputs recorded their sources. The mtime rule is all
        # there is; judging these stale would reconvert every table on upgrade.
        return True
    try:
        return json.loads(recorded) == _source_names(sources)
    except json.JSONDecodeError:
        return False


def _is_up_to_date(group: ConversionGroup) -> bool:
    """True when the output exists, is as new as every source, and was built from exactly them."""
    if not group.out_path.exists():
        return False
    out_mtime = group.out_path.stat().st_mtime
    if not all(f.stat().st_mtime <= out_mtime for f in group.sources):
        return False
    return _built_from(group.out_path, group.sources)


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
        except Exception as e:  # noqa: BLE001 - one group's failure is counted, not allowed to stop the rest
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


def _retire_parquet(dataset: str, workspace: Workspace) -> int:
    """Remove only foehn-authored rolling outputs after an authoritative refresh."""
    candidates = {
        tuple(path.stem.removeprefix(f"{dataset}_").split("_")): path
        for path in workspace.parquet(dataset).glob(f"{dataset}_*.parquet")
    }
    failed = 0
    for group in retired_groups(workspace.bronze(dataset), dataset, candidates):
        path = candidates[group]
        try:
            sources = json.loads(pl.read_parquet_metadata(path).get(_SOURCES_KEY, "null"))
        except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
            # Not a failure: its source is gone, so no run could ever repair it,
            # and counting it would hold the Dataset's cursor back for good.
            logger.warning("  Keeping retired output %s: cannot read its source footer (%s)", path.name, exc)
            continue
        # An unrelated/legacy file without our source footer is not ours to remove.
        if not (isinstance(sources, list) and sources):
            continue
        try:
            path.unlink()
        except OSError as exc:
            failed += 1
            logger.warning("  Could not remove retired output %s: %s", path.name, exc)
        else:
            logger.info("  Removed retired rolling output %s", path.name)
    return failed


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
                lazy_frames = [scan_standard_csv(f, schema_overrides=overrides) for f in group.sources]
                combined = combine_frames(lazy_frames, collection_key)
                combined = derive_timestamp(combined, collection_key)
                _write_parquet_atomic(combined, group.out_path, sources=group.sources)
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
    forecast_local's per-parameter files are joined instead: its newest run
    becomes one row per point and time, every parameter a column.

    Args:
        dataset: Key from COLLECTIONS (e.g. "smn").
        workspace: Where the CSVs are read from and the Parquet is written.

    Returns:
        Number of groups that failed to convert. Zero means everything succeeded;
        non-zero lets callers gate downstream state writes (e.g. ``_last_run.json``).
    """
    csv_dir = workspace.bronze(dataset)
    out_dir = workspace.parquet(dataset)
    failed = _retire_parquet(dataset, workspace)

    groups = [
        ConversionGroup(out_dir / _group_out_name(dataset, group_key), files)
        for group_key, files in group_csv_files(csv_dir, dataset).items()
    ]
    if not groups:
        return failed

    # Load parameter type info from metadata once for the whole collection.
    metadata_types = load_metadata_types(csv_dir)
    return failed + run_conversions(groups, _convert_standard_group(dataset, metadata_types), label=dataset)


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
            if indoor_station(f.name) is None:
                # The archive ships a metadata CSV alongside the data; not a failure.
                skipped += 1
                logger.info("  Skipping non-data file: %s", f.name)
                continue
            frames.append(scan_indoor_csv(f))
        if not frames:
            return 0
        _write_parquet_atomic(pl.concat(frames, how="diagonal_relaxed"), group.out_path, sources=group.sources)
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
            except Exception as e:  # noqa: BLE001 - every unreadable source is counted and named
                # Every unreadable file is reported, not just the first, so one
                # run names all of them.
                failed += 1
                logger.warning("  FAIL %s: %s", f.name, e)
        if failed:
            # All or nothing. Publishing the files that parsed replaced a
            # complete output with a partial one that was then newer than every
            # source, so the next run skipped it as up to date and reported
            # success with the unreadable stations still missing. Leaving the
            # last complete output in place keeps the sources newer than it, so
            # the next run tries again.
            logger.warning(
                "  %s not written: %d of %d source(s) unreadable; the previous output is kept",
                group.out_path.name,
                failed,
                len(group.sources),
            )
            return failed
        _write_parquet_atomic(pl.concat(frames, how="diagonal_relaxed"), group.out_path, sources=group.sources)
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
        df = read_normals_txt(source)
        _write_parquet_atomic(df, group.out_path, compression="snappy", sources=group.sources)
        logger.info("  %s... Converted", source.name)
        return 0

    out_dir = workspace.parquet(dataset)
    groups = [ConversionGroup(out_dir / txt.with_suffix(".parquet").name, [txt]) for txt in txt_files]
    return run_conversions(groups, convert_one, label=dataset)
