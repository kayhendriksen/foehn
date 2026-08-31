"""The three tabular read paths, and the query they all take.

``download`` and ``convert`` route through :mod:`foehn.registry`; loading used to
route through an if-ladder in ``api`` instead, because hoisting the readers into
the registry would have inverted ``registry → api``. The fix was not to hoist but
to drop: these readers depend on ``assets``, ``client``, ``meteocsv`` and
``fetch``, never on ``api``, so they sit *below* the registry and
:class:`~foehn.registry.KindSpec` can carry a ``load`` adapter beside its
``download`` and ``convert`` ones. All three pipeline stages now route the
same way.

Each :class:`Reader` fetches, parses and then finishes its frame: the filters
that apply uniformly — ``drop_null``, ``columns``, ``sort``, ``limit`` and the
calendar predicates — run in :meth:`Reader.finish`. They used to run in
``api.load`` instead, which meant ``registry.load`` handed back a half-filtered
frame and the two schema facts that pass needs, ``key_columns`` and
``sort_column``, sat on ``KindSpec`` where only ``api`` ever read them.
"""

from __future__ import annotations

import io
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import polars as pl

from foehn._urls import asset_filename
from foehn.assets import assets_of, collection_assets, hrefs, select
from foehn.client import check_zip_size
from foehn.collections import COLLECTIONS, DatasetKind, kind
from foehn.fetch import DEFAULT_WORKERS, Fetcher
from foehn.meteocsv import (
    add_indoor_columns,
    decode_meteoswiss_csv,
    derive_timestamp,
    parse_climate_scenarios_csv,
    parse_csv_bytes,
    parse_indoor_filename,
    parse_metadata_types,
    source_columns_for,
    utf8_meteoswiss_csv,
)

# A ``date_to`` of exactly "YYYY-MM-DD" names a whole day, not its midnight.
_BARE_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class Filters:
    """One load query, normalised.

    The eleven filter arguments used to be spelled out at every level: the public
    ``load``, each of the three readers, both shared filter passes, the CLI's
    kwargs ladder and both MCP tools. Packing them once — behind the public
    signature, which is unchanged — means the readers take one parameter, and the
    station/granularity normalisation that was written out three times happens
    here instead.
    """

    stations: frozenset[str] | None = None
    """Lowercased station abbreviations, or None for no filter."""

    granularities: frozenset[str] | None = None
    """Lowercased granularity codes, or None for no filter."""

    time_slices: tuple[str, ...] = ("recent",)
    year: tuple[int, ...] | None = None
    month: tuple[int, ...] | None = None
    date_from: str | None = None
    date_to: str | None = None
    columns: tuple[str, ...] | None = None
    drop_null: str | None = None
    sort: str | None = None
    limit: int | None = None
    workers: int = DEFAULT_WORKERS

    @classmethod
    def build(
        cls,
        *,
        station: str | list[str] | None = None,
        frequency: str | list[str] | None = None,
        time_slice: str | list[str] | None = None,
        year: int | list[int] | None = None,
        month: int | list[int] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        columns: list[str] | None = None,
        drop_null: str | None = None,
        sort: str | None = None,
        limit: int | None = None,
        workers: int = DEFAULT_WORKERS,
    ) -> Filters:
        """Normalise the public keyword arguments into one value.

        An empty list means "no filter", not "match nothing" — the latter is
        never what a caller wants, and the MCP layer used to strip empty lists on
        ``load()``'s behalf. The archive and preamble readers used to read an
        empty list the other way and error out; they now agree with the main path.
        """
        return cls(
            stations=_lower_set(station),
            granularities=_lower_set(frequency),
            time_slices=_as_tuple(time_slice) or ("recent",),
            year=_as_tuple(year),
            month=_as_tuple(month),
            date_from=date_from,
            date_to=date_to,
            columns=tuple(columns) if columns else None,
            drop_null=drop_null,
            sort=sort,
            limit=limit,
            workers=workers,
        )

    @property
    def has_calendar_filter(self) -> bool:
        """Whether any of the calendar predicates was asked for."""
        return any(x is not None for x in (self.year, self.month, self.date_from, self.date_to))


def _lower_set(value: str | list[str] | None) -> frozenset[str] | None:
    if not value:
        return None
    return frozenset({value.lower()} if isinstance(value, str) else {v.lower() for v in value})


def _as_tuple(value):
    if value is None:
        return None
    if isinstance(value, (str, int)):
        return (value,)
    return tuple(value) or None


class ReadAdapter(Protocol):
    """How one :class:`~foehn.collections.DatasetKind` fetches and parses its frame."""

    def __call__(self, dataset: str, filters: Filters, *, fetcher: Fetcher) -> pl.DataFrame: ...


def _require_columns(df: pl.DataFrame, names: list[str], label: str) -> None:
    """Raise ValueError naming any of *names* the loaded frame doesn't have.

    Silently ignoring an unknown column turns a mistyped MeteoSwiss shortcode
    (``tre200dO`` for ``tre200d0``) into a plausible-looking wrong answer: the
    ``columns`` filter returns only the always-kept key columns, and
    ``drop_null`` keeps every null row it was asked to remove. Both are worth an
    error, especially on the MCP surface where the caller is an LLM guessing
    parameter names.
    """
    missing = [n for n in names if n not in df.columns]
    if not missing:
        return
    available = ", ".join(sorted(df.columns))
    raise ValueError(f"Unknown column(s) {missing} in {label}=. This dataset has: {available}")


@dataclass(frozen=True)
class Reader:
    """How one tabular :class:`~foehn.collections.DatasetKind` becomes a DataFrame.

    Constructed in :data:`~foehn.registry.KINDS`, exactly as
    :class:`~foehn.grids.GridReader` is. ``key_columns`` and ``sort_column`` used
    to sit on :class:`~foehn.registry.KindSpec` instead — they are facts about
    the frame a reader produces, and they lived a layer above the reader only
    because the post-filter did. Both are here now, so the reader finishes its
    own frame and ``registry.load`` hands back a complete one.
    """

    read: ReadAdapter
    """Fetches and parses. The only part that differs between kinds."""

    key_columns: tuple[str, ...] = ("station_abbr", "reference_timestamp")
    """Columns an explicit ``columns=`` selection always keeps."""

    sort_column: str = "reference_timestamp"
    """What ``sort=`` orders by. The nominal-date kind has no real timestamp."""

    def finish(self, df: pl.DataFrame, filters: Filters) -> pl.DataFrame:
        """Apply the row/column filters that are the same for every kind.

        Uniform, which is why it is one method rather than part of each
        ``read``: ``read_standard`` applies the time filters per CSV to bound
        peak memory, and everything else happens once, here, on the concatenated
        frame. ``drop_null`` deliberately runs only here — a frame missing that
        column keeps every row, while after a diagonal concat the column exists
        as null across those rows and they are dropped.
        """
        df = apply_time_filters(df, filters)
        if filters.drop_null:
            _require_columns(df, [filters.drop_null], "drop_null")
            df = df.filter(pl.col(filters.drop_null).is_not_null())
        if filters.sort in ("asc", "desc"):
            df = df.sort(self.sort_column, descending=(filters.sort == "desc"))
        if filters.columns:
            _require_columns(df, list(filters.columns), "columns")
            keep = [c for c in self.key_columns if c in df.columns]
            keep += [c for c in filters.columns if c not in keep]
            df = df.select(keep)
        if filters.limit is not None:
            df = df.head(filters.limit)
        return df


# --- Shared row predicates ---


def apply_time_filters(df: pl.DataFrame, filters: Filters) -> pl.DataFrame:
    """Apply the timestamp row predicates, on one parsed CSV or on the concatenation.

    ``read_standard`` runs these on each CSV as it is parsed, instead of only on
    the concatenated result. These are all per-row predicates, so filtering early
    is equivalent — but it bounds peak memory by the largest single station file
    rather than by the whole matched set. ``drop_null`` deliberately stays in the
    post-filter: a frame missing that column keeps every row here, while after a
    diagonal concat the column exists as null across those rows and they are
    dropped.
    """
    ts = "reference_timestamp"
    if filters.year is not None:
        df = df.filter(pl.col(ts).dt.year().is_in(list(filters.year)))
    if filters.month is not None:
        df = df.filter(pl.col(ts).dt.month().is_in(list(filters.month)))
    # Cast the timestamp column to Datetime before comparing: some daily/monthly
    # files parse ``reference_timestamp`` as a Date, and comparing Date vs the
    # Datetime literal would raise. Date→Datetime and Datetime→Datetime are both safe.
    if filters.date_from is not None:
        df = df.filter(pl.col(ts).cast(pl.Datetime) >= pl.lit(filters.date_from).str.to_datetime())
    if filters.date_to is not None:
        bound = pl.lit(filters.date_to).str.to_datetime()
        if _BARE_DATE_RE.match(filters.date_to):
            # A bare "YYYY-MM-DD" means the whole of that day. Comparing <= the
            # parsed midnight is right for d/m/y (timestamps sit at 00:00) but
            # silently drops every 10-minute and hourly reading after 00:00, so
            # bound the day exclusively at the next midnight instead.
            df = df.filter(pl.col(ts).cast(pl.Datetime) < bound.dt.offset_by("1d"))
        else:
            df = df.filter(pl.col(ts).cast(pl.Datetime) <= bound)
    return df


def _fetch_frames(fetch_one, targets: list[str], workers: int) -> list[pl.DataFrame]:
    """Parse each target concurrently.

    The fetcher is safe to share across the pool: it hands each worker thread its
    own session.
    """
    if len(targets) == 1 or workers <= 1:
        return [fetch_one(t) for t in targets]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fetch_one, targets))


# --- The readers ---


def read_standard(dataset: str, filters: Filters, *, fetcher: Fetcher) -> pl.DataFrame:
    """Per-station CSVs split by time slice and granularity — the main path.

    Also serves the forecast kind, whose filenames carry no time slice: there the
    newest run is what bounds the set instead.
    """
    collection_id = COLLECTIONS[dataset]

    # 1. Fetch metadata types for schema inference.
    metadata_types: dict[str, type[pl.DataType]] = {}
    coll = fetcher.collection(collection_id)
    for asset in collection_assets(coll, suffixes=(".csv",), contains="_meta_parameters"):
        metadata_types = parse_metadata_types(decode_meteoswiss_csv(fetcher.get(asset.href, timeout=60).body))
        break

    # 2. Get STAC items and collect matching CSV URLs.
    items = fetcher.items(collection_id)

    # Filter items by station (item id = station abbreviation).
    if filters.stations is not None:
        items = [item for item in items if item.get("id", "").lower() in filters.stations]

    # A forecast item is one *day*, not one forecast, and the newest one is empty
    # until that day's runs publish — so keep all items and narrow to the newest
    # run by filename below. Ranking on ``datetime`` would not help either: it is a
    # refresh timestamp, identical across items to the microsecond.
    is_forecast = kind(dataset) is DatasetKind.FORECAST_CSV
    if is_forecast and items:
        items.sort(key=lambda x: x.get("id", ""))

    csv_hrefs = hrefs(
        select(
            assets_of(items, suffixes=(".csv",)),
            time_slices=None if is_forecast else list(filters.time_slices),
            granularities=filters.granularities,
            latest_run=is_forecast,
        )
    )

    if not csv_hrefs:
        described = (
            f"station={sorted(filters.stations) if filters.stations else None}, "
            f"frequency={sorted(filters.granularities) if filters.granularities else None}, "
            f"time_slice={list(filters.time_slices)}"
        )
        raise ValueError(f"No CSV files found for {dataset!r} with {described}.")

    # 3. Download and parse each CSV concurrently.
    # With an explicit ``columns=``, tell the parser up front instead of parsing all
    # ~42 columns of every station file and selecting afterwards. The frame retained
    # per station drops by an order of magnitude, which is what bounds peak memory
    # while the whole matched set is assembled. Everything the later filters and the
    # concat rely on has to survive the projection.
    wanted_columns: set[str] | None = None
    if filters.columns:
        wanted_columns = {"station_abbr", "reference_timestamp", *filters.columns}
        if filters.drop_null:
            wanted_columns.add(filters.drop_null)
        # Whatever this kind derives its timestamp from has to survive the projection.
        wanted_columns |= source_columns_for(dataset)

    def _fetch(href: str) -> pl.DataFrame:
        # Zero-copy when the payload is already UTF-8 (the usual case): these are
        # the big files, and ``workers`` of them are in flight at once.
        body = fetcher.get(href, timeout=60).body
        frame = parse_csv_bytes(utf8_meteoswiss_csv(body), metadata_types, wanted_columns=wanted_columns)
        # Drop the rows this call can never return *before* they reach the
        # concat. Every frame is otherwise held in full until the whole matched
        # set is materialised, so a narrow year= over many stations peaked at the
        # size of the entire time slice. A kind that derives its timestamp gets
        # it here, so its frames can be narrowed too.
        frame = derive_timestamp(frame, dataset)
        if "reference_timestamp" in frame.columns:
            frame = apply_time_filters(frame, filters)
        return frame

    df = pl.concat(_fetch_frames(_fetch, csv_hrefs, filters.workers), how="diagonal_relaxed")

    # Again on the concatenation: a station whose file was empty contributes no
    # rows above, and the diagonal concat can still leave the column absent.
    return derive_timestamp(df, dataset)


def read_preamble(dataset: str, filters: Filters, *, fetcher: Fetcher) -> pl.DataFrame:
    """CH2025 climate-scenario CSVs (metadata preamble + wide model table).

    Dates are nominal (0001..0030 on a 365-day calendar), so the calendar-based
    year/month/date filters do not apply here — they are rejected before this is
    reached; ``sort`` orders lexically by the string ``date`` column.
    """
    items = fetcher.items(COLLECTIONS[dataset])
    if filters.stations is not None:
        items = [item for item in items if item.get("id", "").lower() in filters.stations]

    csv_hrefs = hrefs(assets_of(items, suffixes=(".csv",), excludes="_meta_"))
    if not csv_hrefs:
        stations = sorted(filters.stations) if filters.stations else None
        raise ValueError(f"No climate-scenario CSVs found for {dataset!r} with station={stations}.")

    def _fetch(href: str) -> pl.DataFrame:
        return parse_climate_scenarios_csv(fetcher.get(href, timeout=120).body, asset_filename(href))

    return pl.concat(_fetch_frames(_fetch, csv_hrefs, filters.workers), how="diagonal_relaxed")


def read_archive(dataset: str, filters: Filters, *, fetcher: Fetcher) -> pl.DataFrame:
    """A single ZIP of per-station CSVs (indoor scenarios).

    Unlike the per-station collections, this is a single archive, so the whole
    ZIP is fetched and parsed in memory; the station filter picks which member
    CSVs are parsed.
    """
    items = fetcher.items(COLLECTIONS[dataset])
    archives = assets_of(items, suffixes=(".zip",))
    if not archives:
        raise ValueError(f"No .zip asset found for {dataset!r}.")
    zip_href = archives[0].href

    archive = fetcher.get(zip_href, timeout=300).body

    frames: list[pl.DataFrame] = []
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        # Everything below is parsed in memory — refuse a decompression bomb.
        check_zip_size(zf, asset_filename(zip_href))
        for name in zf.namelist():
            if not name.endswith(".csv"):
                continue
            parsed = parse_indoor_filename(Path(name).stem)
            if parsed is None:
                continue
            st, period, scenario, variant = parsed
            if filters.stations is not None and st.lower() not in filters.stations:
                continue
            with zf.open(name) as fh:
                frame = pl.read_csv(fh.read(), separator=",", infer_schema_length=10_000, truncate_ragged_lines=True)
            frames.append(add_indoor_columns(frame, st, period, scenario, variant))

    if not frames:
        stations = sorted(filters.stations) if filters.stations else None
        raise ValueError(f"No indoor data found for {dataset!r} with station={stations}.")

    return pl.concat(frames, how="diagonal_relaxed")


__all__ = [
    "Filters",
    "ReadAdapter",
    "Reader",
    "apply_time_filters",
    "read_archive",
    "read_preamble",
    "read_standard",
]
