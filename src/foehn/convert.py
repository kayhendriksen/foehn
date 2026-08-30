"""Convert downloaded CSVs and TXTs to Parquet using Polars."""

from __future__ import annotations

import codecs
import io
import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl

from foehn.collections import COLLECTIONS, NO_GRANULARITY_KINDS, kind
from foehn.workspace import Workspace

logger = logging.getLogger(__name__)

_COL_RE = re.compile(r"at column '([^']+)'")


def decode_meteoswiss_csv(content: bytes) -> str:
    """Decode MeteoSwiss CSV bytes to text.

    MeteoSwiss CSVs are usually UTF-8 (often with a BOM) but some legacy files
    are Windows-1252. Try UTF-8 (BOM-aware) first, falling back to Windows-1252.
    The fallback replaces the five bytes cp1252 leaves unmapped (0x81, 0x8D,
    0x8F, 0x90, 0x9D) rather than raising, so the decode is total.
    """
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return content.decode("windows-1252", errors="replace")


_UTF8_BOM = b"\xef\xbb\xbf"

# Validate in 1 MiB steps rather than decoding the whole buffer: the point is to
# confirm the bytes are UTF-8 without ever materialising the file as a str.
_UTF8_VALIDATE_STEP = 1 << 20


def utf8_meteoswiss_csv(content: bytes) -> bytes | memoryview:
    """Return *content* as UTF-8 bytes, re-encoding only when it isn't already.

    :func:`decode_meteoswiss_csv` exists to produce text. Callers that only want
    bytes to hand to Polars were round-tripping through it — bytes → str →
    bytes — which keeps three copies of the payload alive at once. On an 80 MB
    CSV that measured 161 MB of transient allocation per file, and the download
    paths run eight of these concurrently.

    MeteoSwiss CSVs are usually UTF-8 (often BOM-prefixed), so the common path
    validates incrementally and hands back a ``memoryview`` over the caller's
    original buffer: no copy, and the BOM is skipped by slicing rather than by
    building a new bytes object. Only genuine Windows-1252 files pay for a
    re-encode. The returned view borrows *content*, which must outlive it.
    """
    view = memoryview(content)
    has_bom = content.startswith(_UTF8_BOM)
    if has_bom:
        view = view[len(_UTF8_BOM) :]

    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        for i in range(0, len(view), _UTF8_VALIDATE_STEP):
            decoder.decode(view[i : i + _UTF8_VALIDATE_STEP])
        decoder.decode(b"", final=True)
    except UnicodeDecodeError:
        return content.decode("windows-1252", errors="replace").encode("utf-8")
    # Hand back the caller's own bytes when there is nothing to strip. A
    # memoryview is only needed to skip a BOM without copying, and it is the
    # costlier return: io.BytesIO(bytes) shares the buffer, but wrapping a
    # memoryview copies the whole payload.
    return content if not has_bom else view


_DTYPE_MAP: dict[str, type[pl.DataType]] = {
    "float": pl.Float64,
    "integer": pl.Int64,
}


def parse_metadata_types(content: bytes | str) -> dict[str, type[pl.DataType]]:
    """Build a parameter→Polars dtype mapping from metadata CSV content.

    Works with both raw bytes (in-memory) and string content.
    Returns an empty dict if the expected columns are missing.

    Public because the in-memory load path crosses this seam for it: schema
    inference from a collection's ``_meta_parameters.csv`` is one rule, and a
    name-mangled import was hiding that it is part of this module's interface.
    """
    try:
        if isinstance(content, str):
            content = content.encode("utf-8")
        meta = pl.read_csv(io.BytesIO(content), separator=";", infer_schema_length=0)
    except Exception:
        return {}

    if "parameter_shortname" not in meta.columns or "parameter_datatype" not in meta.columns:
        return {}

    type_map: dict[str, type[pl.DataType]] = {}
    for row in meta.select("parameter_shortname", "parameter_datatype").iter_rows():
        shortname, datatype = row
        if shortname and datatype:
            dtype = _DTYPE_MAP.get(datatype.strip().lower())
            if dtype is not None:
                type_map[shortname] = dtype
    return type_map


def load_metadata_types(csv_dir: Path) -> dict[str, type[pl.DataType]]:
    """Build a parameter→Polars dtype mapping from a *_meta_parameters.csv file.

    Returns an empty dict if no metadata file is found or if the expected
    columns (``parameter_shortname``, ``parameter_datatype``) are missing.

    The file-based counterpart to :func:`parse_metadata_types`, and public for
    the same reason: the Databricks ingest script reads it across this seam.
    """
    # sorted(): glob order is filesystem-dependent, so an unsorted [0] picks
    # arbitrarily when a collection ships more than one metadata file.
    meta_files = sorted(csv_dir.glob("*_meta_parameters.csv"))
    if not meta_files:
        return {}

    meta_path = meta_files[0]
    try:
        return parse_metadata_types(meta_path.read_bytes())
    except Exception:
        return {}


def parse_csv_bytes(
    content: bytes | memoryview,
    metadata_types: dict[str, type[pl.DataType]] | None = None,
    wanted_columns: set[str] | None = None,
) -> pl.DataFrame:
    """Parse CSV bytes into a Polars DataFrame, applying metadata type overrides.

    Args:
        content: Raw CSV bytes (UTF-8 encoded); a memoryview is accepted so
            callers can pass a zero-copy view from ``utf8_meteoswiss_csv``.
        metadata_types: Optional parameter→dtype mapping from metadata.
        wanted_columns: Only parse these columns (intersected with the file's own
            header, so a station missing one is not an error — the diagonal concat
            fills it with nulls exactly as before). A station file is ~42 columns
            and a typical query wants two or three, so skipping the rest cuts both
            parse time and, far more importantly, the frame retained per station
            while the whole matched set is assembled.

    Returns:
        Parsed Polars DataFrame.
    """
    # Polars reads a bytes object without copying it and strips any UTF-8 BOM
    # itself; a memoryview has to be materialised once, here rather than per read.
    data = content if isinstance(content, bytes) else bytes(content)

    header: list[str] | None = None
    if metadata_types or wanted_columns:
        try:
            # Parse the header line alone — wrapping the whole payload to read one
            # line costs a copy of the entire file.
            end = data.find(b"\n")
            header = pl.read_csv(
                data[: end + 1] if end != -1 else data, separator=";", n_rows=0, infer_schema_length=0
            ).columns
        except Exception as exc:
            logger.debug("Could not read CSV header (%s) — parsing every column, inferring types", exc)

    # Intersect in header order. Falling back to "everything" when nothing matches
    # keeps read_csv(columns=[]) — which is an error — off the table.
    use_columns: list[str] | None = None
    if wanted_columns and header is not None:
        use_columns = [c for c in header if c in wanted_columns] or None

    # Build per-file overrides by matching CSV columns to metadata types. Restricted
    # to the columns actually being read: polars rejects an override naming a column
    # that the projection excluded.
    overrides: dict[str, type[pl.DataType]] = {}
    if metadata_types and header is not None:
        for col in use_columns if use_columns is not None else header:
            if col in metadata_types:
                overrides[col] = metadata_types[col]

    try:
        return pl.read_csv(
            data,
            separator=";",
            infer_schema_length=100,
            try_parse_dates=True,
            schema_overrides=overrides or None,
            columns=use_columns,
        )
    except (pl.exceptions.ComputeError, pl.exceptions.SchemaError) as e:
        # Fallback: accumulate Float64 overrides for problematic columns.
        last_err = e
        while True:
            m = _COL_RE.search(str(last_err))
            # A column already forced to Float64 can't be widened further —
            # bail out instead of retrying the same parse forever.
            if not m or overrides.get(m.group(1)) == pl.Float64:
                break
            if use_columns is not None and m.group(1) not in use_columns:
                break  # not a column we're reading; widening it would be rejected
            overrides[m.group(1)] = pl.Float64
            try:
                return pl.read_csv(
                    data,
                    separator=";",
                    infer_schema_length=100,
                    try_parse_dates=True,
                    schema_overrides=overrides,
                    columns=use_columns,
                )
            except (pl.exceptions.ComputeError, pl.exceptions.SchemaError) as e2:
                last_err = e2
            except Exception:
                raise
        raise last_err from None


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
"""Build ``group.out_path`` from ``group.sources``; return source-level failures tolerated.

A converter that skips an unreadable source file and writes the rest reports how
many it skipped, so the caller's exit code still reflects them. Raising instead
means the whole group failed.
"""


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


def group_csv_files(csv_dir: Path, collection_key: str) -> dict[tuple[str, ...], list[Path]]:
    """Group a collection's data CSVs by (frequency, time_slice).

    MeteoSwiss names standard CSV assets
    ``ogd-{key}_{station}_{granularity}[_{timeslice}].csv``, so the group key is
    read out of the filename: ``ogd-smn_ber_d_recent.csv`` → ``("d", "recent")``.
    Metadata CSVs are excluded, and collections whose filenames carry no
    granularity segment collapse into a single unkeyed group.

    Shared with the Delta ingestion script, which builds one table per group and
    must agree with the Parquet converter on where the boundaries are — these
    are upstream naming rules, and two copies of them drift.
    """
    csv_files = [f for f in sorted(csv_dir.glob("*.csv")) if "_meta_" not in f.name]
    groups: dict[tuple[str, ...], list[Path]] = {}

    # No granularity in the filename at all (forecast_local's vnut12.lssw.* names,
    # climate_scenarios). Returning early also avoids slicing off a prefix these
    # names do not carry, which would chop arbitrary characters off the stem.
    if kind(collection_key) in NO_GRANULARITY_KINDS:
        if csv_files:
            groups[()] = csv_files
        return groups

    # Derive the filename prefix from the collection ID (e.g. "ogd-smn").
    prefix = COLLECTIONS[collection_key].rsplit(".", 1)[-1]
    for csv_path in csv_files:
        suffix_part = csv_path.stem[len(prefix) + 1 :]  # e.g. "ber_d_recent"
        parts = suffix_part.split("_")
        if len(parts) > 2:
            group_key = (parts[1], parts[2])  # (frequency, time_slice)
        else:
            group_key = (parts[1],) if len(parts) > 1 else ()  # (frequency,)
        groups.setdefault(group_key, []).append(csv_path)

    return groups


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


_CS_HEADER_PREFIX = "DATE;"


def parse_climate_scenarios_filename(filename: str) -> tuple[str, str, str]:
    """Parse a climate-scenario filename into (station, variable, gwl).

    Files are named ``ogd-climate-scenarios-ch2025_{station}_{variable}_{gwl}``.
    """
    stem = filename.rsplit("/", 1)[-1]
    stem = stem.removesuffix(".csv")
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(
            f"Unexpected climate-scenario filename {filename!r}: expected "
            "'..._{station}_{variable}_{gwl}', cannot extract station/variable/gwl."
        )
    return parts[-3], parts[-2], parts[-1]


def add_climate_scenarios_columns(frame, station: str, variable: str, gwl: str):
    """Add filename-derived columns and move the key columns to the front.

    Works on both a LazyFrame (scan_csv) and a DataFrame (read_csv)."""
    frame = frame.rename({"DATE": "date"}).with_columns(
        pl.lit(station).alias("station_abbr"),
        pl.lit(variable).alias("variable"),
        pl.lit(gwl).alias("gwl"),
    )
    cols = frame.collect_schema().names() if isinstance(frame, pl.LazyFrame) else frame.columns
    front = ["station_abbr", "variable", "gwl", "date"]
    return frame.select(front + [c for c in cols if c not in front])


def parse_climate_scenarios_csv(content: bytes | str, filename: str) -> pl.DataFrame:
    """Parse a CH2025 climate-scenario CSV into a wide model table.

    These files carry a multi-row ``KEY;VALUE`` metadata preamble before the
    real ``DATE;<model>;<model>;...`` header. We skip the preamble, read the
    table, and tag rows with station/variable/gwl parsed from the filename
    (``ogd-climate-scenarios-ch2025_{station}_{variable}_{gwl}``). The DATE
    column is kept as a string because it encodes a nominal 30-year period
    (0001-01-01 … 0030-12-31 on a 365-day calendar), not real calendar dates.

    This is the in-memory path (used when streaming a download); the file-based
    converter uses :func:`scan_climate_scenarios_csv` instead.
    """
    text = content.decode("utf-8-sig", errors="replace") if isinstance(content, bytes) else content
    lines = text.splitlines()
    header_idx = next((i for i, line in enumerate(lines) if line.startswith(_CS_HEADER_PREFIX)), None)
    if header_idx is None:
        raise ValueError(f"No 'DATE;' data header found in {filename!r}")

    station, variable, gwl = parse_climate_scenarios_filename(filename)

    table = "\n".join(lines[header_idx:])
    df = pl.read_csv(
        io.BytesIO(table.encode("utf-8")),
        separator=";",
        infer_schema_length=20_000,
        truncate_ragged_lines=True,
    )
    return add_climate_scenarios_columns(df, station, variable, gwl)


def _climate_scenarios_preamble_lines(path: Path) -> int:
    """Count the metadata preamble lines before the ``DATE;`` header.

    Reads the file line by line and stops at the header, so a file of any size
    costs only its preamble — the whole point of not slurping it into memory.
    """
    with path.open("rb") as fh:
        for i, raw in enumerate(fh):
            if raw.decode("utf-8-sig", errors="replace").startswith(_CS_HEADER_PREFIX):
                return i
    raise ValueError(f"No 'DATE;' data header found in {path.name!r}")


def scan_climate_scenarios_csv(path: Path) -> pl.LazyFrame:
    """Lazily scan a CH2025 climate-scenario CSV, skipping its metadata preamble.

    ``skip_lines`` (not ``skip_rows``) matches how the preamble is counted: raw
    newlines, ignoring CSV quoting, so a stray quote in a metadata value can't
    shift the header offset.
    """
    station, variable, gwl = parse_climate_scenarios_filename(path.name)
    lf = pl.scan_csv(
        path,
        separator=";",
        skip_lines=_climate_scenarios_preamble_lines(path),
        infer_schema_length=20_000,
        truncate_ragged_lines=True,
    )
    return add_climate_scenarios_columns(lf, station, variable, gwl)


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
