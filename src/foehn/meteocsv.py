"""How MeteoSwiss writes a CSV, and everything foehn needs to read one back.

The encoding fallback, the semicolon separator, the dtype overrides from a
collection's ``_meta_parameters.csv``, the ``KEY;VALUE`` preamble, and the
filename rules that say which station, granularity, scenario or forecast run a
file holds. Upstream conventions, in one place, below every pipeline that reads
them.

Split out of ``foehn.convert``, which owned these *and* the Bronze → Parquet
stage. Five modules imported that one: four of them — ``readers`` for the load
path, ``transfer`` for one byte-level helper, ``api``, and the Databricks ingest
script, which writes Delta and never produces a Parquet file — wanted only the
conventions, and only ``registry`` wanted the Parquet stage. So the download
engine depended on the Parquet converter, and so, through it, did the gridded
read path. Nothing here knows what a Parquet file is; ``convert`` imports this
module and not the other way round.
"""

from __future__ import annotations

import codecs
import io
import logging
import re
from pathlib import Path

import polars as pl

from foehn.collections import COLLECTIONS, DERIVED_TIMESTAMP_KINDS, NO_GRANULARITY_KINDS, kind

logger = logging.getLogger(__name__)

# Polars names the offending column in its dtype errors. Both readers of a
# MeteoSwiss CSV widen that column to Float64 and retry — the in-memory parse
# here and the streaming convert in ``foehn.convert`` — so the pattern is stated
# once, where the parsing rules live.
_COL_RE = re.compile(r"at column '([^']+)'")


def column_from_dtype_error(message: str) -> str | None:
    """Return the column polars blamed in a dtype error, or None if it named none."""
    found = _COL_RE.search(message)
    return found.group(1) if found else None


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
            col = column_from_dtype_error(str(last_err))
            # A column already forced to Float64 can't be widened further —
            # bail out instead of retrying the same parse forever.
            if col is None or overrides.get(col) == pl.Float64:
                break
            if use_columns is not None and col not in use_columns:
                break  # not a column we're reading; widening it would be rejected
            overrides[col] = pl.Float64
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


def scan_standard_csv(path: Path, *, schema_overrides: dict[str, type[pl.DataType]] | None = None):
    """Lazily scan one standard MeteoSwiss CSV, without reading it.

    The lazy half of :func:`parse_csv_bytes`, which is the eager one. The convert
    stage used to spell these options out itself — the last kind whose conventions
    were stated above this module — because the dtype-drift retry is wrapped
    around the scan. That retry is the convert stage's: it re-runs the sink, and
    passes the widened types back in through *schema_overrides*.

    A wider schema window than the eager path's, on purpose: a scan pays for the
    inferred rows once per file rather than holding the file, and the converter's
    groups are whole historical series.
    """
    return pl.scan_csv(
        path,
        separator=";",
        try_parse_dates=True,
        schema_overrides=schema_overrides or None,
        infer_schema_length=10_000,
    )


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


def derive_timestamp(frame, dataset: str):
    """Give *frame* a ``reference_timestamp`` if its kind has to derive one.

    A no-op for every kind whose files carry the column already, so callers make
    the call unconditionally instead of asking which dataset they hold. Works on
    a LazyFrame or a DataFrame.
    """
    if kind(dataset) not in DERIVED_TIMESTAMP_KINDS:
        return frame
    return add_forecast_local_timestamp(frame)


def source_columns_for(dataset: str) -> set[str]:
    """Extra source columns :func:`derive_timestamp` needs, beyond what was asked for.

    An explicit ``columns=`` narrows the parse to bound memory, so anything the
    derivation reads has to survive that projection — dropping ``Date`` leaves a
    forecast frame with no timestamp to filter or sort on.
    """
    return {"Date"} if kind(dataset) in DERIVED_TIMESTAMP_KINDS else set()


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


_INDOOR_TIME_COLS = ["time.yy", "time.mm", "time.dd", "time.hh"]

# How MeteoSwiss writes the indoor archive's member CSVs: comma-separated, with
# the timestamp split across four integer columns. Stated once, for both the
# eager read and the lazy scan — the load path and the convert stage each used
# to spell them out.
_INDOOR_CSV_OPTIONS = {"separator": ",", "infer_schema_length": 10_000, "truncate_ragged_lines": True}


def indoor_station(filename: str) -> str | None:
    """Whose data an indoor archive member is, or None if it is not data at all.

    Data files are ``{station}_{period}_{scenario}_{variant}`` with a 4-digit
    year as the period; the archive ships a metadata CSV alongside them. Both
    callers ask this before reading — the load path to skip a station it was not
    asked for, the convert stage to count what it skipped — so it is one
    question, asked of the name, and not a by-product of parsing the file.
    """
    parsed = _parse_indoor_filename(filename)
    return None if parsed is None else parsed[0]


def _parse_indoor_filename(filename: str) -> tuple[str, str, str, str] | None:
    """Split a member's stem into (station, period, scenario, variant), or None."""
    parts = Path(filename).name.removesuffix(".csv").split("_")
    if len(parts) < 4 or not parts[1].isdigit():
        return None
    return parts[0], parts[1], parts[2], "_".join(parts[3:])


def parse_indoor_csv(content: bytes, filename: str) -> pl.DataFrame:
    """Read one indoor archive member from memory, tagged from its filename.

    The in-memory path, used when the ZIP is streamed rather than extracted; the
    converter uses :func:`scan_indoor_csv`. Both are given a member the caller
    has already accepted via :func:`indoor_station`.
    """
    return _add_indoor_columns(pl.read_csv(content, **_INDOOR_CSV_OPTIONS), _indoor_identity(filename))


def scan_indoor_csv(path: Path) -> pl.LazyFrame:
    """Lazily scan one extracted indoor archive member, tagged from its filename."""
    return _add_indoor_columns(pl.scan_csv(path, **_INDOOR_CSV_OPTIONS), _indoor_identity(path.name))


def _indoor_identity(filename: str) -> tuple[str, str, str, str]:
    parsed = _parse_indoor_filename(filename)
    if parsed is None:
        raise ValueError(f"{filename!r} is not an indoor data CSV — check indoor_station() before reading it.")
    return parsed


def _add_indoor_columns(frame, identity: tuple[str, str, str, str]):
    """Add reference_timestamp + filename-derived columns and drop the raw time
    columns. Works on both a LazyFrame (scan_csv) and a DataFrame (read_csv)."""
    station, period, scenario, variant = identity
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


_CS_HEADER_PREFIX = "DATE;"


def _parse_climate_scenarios_filename(filename: str) -> tuple[str, str, str]:
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


def _add_climate_scenarios_columns(frame, station: str, variable: str, gwl: str):
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

    station, variable, gwl = _parse_climate_scenarios_filename(filename)

    table = "\n".join(lines[header_idx:])
    df = pl.read_csv(
        io.BytesIO(table.encode("utf-8")),
        separator=";",
        infer_schema_length=20_000,
        truncate_ragged_lines=True,
    )
    return _add_climate_scenarios_columns(df, station, variable, gwl)


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
    station, variable, gwl = _parse_climate_scenarios_filename(path.name)
    lf = pl.scan_csv(
        path,
        separator=";",
        skip_lines=_climate_scenarios_preamble_lines(path),
        infer_schema_length=20_000,
        truncate_ragged_lines=True,
    )
    return _add_climate_scenarios_columns(lf, station, variable, gwl)
