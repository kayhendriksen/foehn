"""The three tabular read paths, and the query they all take.

``download`` and ``convert`` route through :mod:`foehn.registry`; loading used to
route through an if-ladder in ``api`` instead, because hoisting the readers into
the registry would have inverted ``registry → api``. The fix was not to hoist but
to drop: these readers depend on ``assets``, ``archives``, ``meteocsv`` and
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
from datetime import UTC, datetime
from typing import Protocol

import polars as pl

from foehn._urls import asset_filename
from foehn.archives import check_zip_size
from foehn.assets import assets_of, collection_assets, hrefs, select
from foehn.collections import COLLECTIONS, DEFAULT_TIME_SLICE
from foehn.fetch import DEFAULT_WORKERS, Fetcher
from foehn.meteocsv import (
    decode_meteoswiss_csv,
    derive_timestamp,
    indoor_station,
    parse_climate_scenarios_csv,
    parse_csv_bytes,
    parse_indoor_csv,
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

    time_slices: tuple[str, ...] = (DEFAULT_TIME_SLICE,)
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
        months = _as_tuple(month)
        if months is not None and (invalid_months := sorted({value for value in months if not 1 <= value <= 12})):
            raise ValueError(f"Invalid month {invalid_months}. Valid options: 1 through 12.")
        if sort is not None and sort not in ("asc", "desc"):
            raise ValueError(f"Invalid sort {sort!r}. Valid options: asc, desc.")
        if limit is not None and limit < 0:
            raise ValueError("limit must be zero or greater.")
        if workers < 1:
            raise ValueError("workers must be greater than zero.")
        start = _parse_date_bound(date_from, "date_from")
        end = _parse_date_bound(date_to, "date_to")
        if start is not None and end is not None and start > end:
            raise ValueError("date_from must be before or equal to date_to.")
        return cls(
            stations=_lower_set(station),
            granularities=_lower_set(frequency),
            time_slices=_as_tuple(time_slice) or (DEFAULT_TIME_SLICE,),
            year=_as_tuple(year),
            month=months,
            date_from=_normalized_date_bound(date_from, start),
            date_to=_normalized_date_bound(date_to, end),
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


def _parse_date_bound(value: str | None, label: str) -> datetime | None:
    """Validate an ISO date/datetime and return a comparable UTC-naive value."""
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{label} must be an ISO date or datetime, got {value!r}.") from None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed


def _normalized_date_bound(value: str | None, parsed: datetime | None) -> str | None:
    """Keep whole-day syntax; express datetimes as UTC-naive for Polars."""
    if value is None or parsed is None:
        return None
    if _BARE_DATE_RE.match(value):
        return value
    return parsed.isoformat()


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


def _metadata_types(dataset: str, fetcher: Fetcher) -> dict[str, type[pl.DataType]]:
    """Read the Collection's dtype declarations once for either CSV Reader."""
    collection_id = COLLECTIONS[dataset]
    metadata_types: dict[str, type[pl.DataType]] = {}
    coll = fetcher.collection(collection_id)
    for asset in collection_assets(coll, suffixes=(".csv",), contains="_meta_parameters"):
        metadata_types = parse_metadata_types(decode_meteoswiss_csv(fetcher.get(asset.href, timeout=60).body))
        break
    return metadata_types


def _read_csv_hrefs(
    dataset: str,
    filters: Filters,
    csv_hrefs: list[str],
    *,
    fetcher: Fetcher,
    metadata_types: dict[str, type[pl.DataType]],
) -> pl.DataFrame:
    """Fetch and normalize already-selected Standard or Forecast CSV Assets."""
    if not csv_hrefs:
        described = (
            f"station={sorted(filters.stations) if filters.stations else None}, "
            f"frequency={sorted(filters.granularities) if filters.granularities else None}, "
            f"time_slice={list(filters.time_slices)}"
        )
        raise ValueError(f"No CSV files found for {dataset!r} with {described}.")

    wanted_columns: set[str] | None = None
    if filters.columns:
        wanted_columns = {"station_abbr", "reference_timestamp", *filters.columns}
        if filters.drop_null:
            wanted_columns.add(filters.drop_null)
        wanted_columns |= source_columns_for(dataset)

    def _fetch(href: str) -> pl.DataFrame:
        body = fetcher.get(href, timeout=60).body
        frame = parse_csv_bytes(utf8_meteoswiss_csv(body), metadata_types, wanted_columns=wanted_columns)
        frame = derive_timestamp(frame, dataset)
        if "reference_timestamp" in frame.columns:
            frame = apply_time_filters(frame, filters)
        return frame

    df = pl.concat(_fetch_frames(_fetch, csv_hrefs, filters.workers), how="diagonal_relaxed")
    return derive_timestamp(df, dataset)


def read_standard(dataset: str, filters: Filters, *, fetcher: Fetcher) -> pl.DataFrame:
    """Per-station CSVs split by Time slice and Granularity — the main path."""
    collection_id = COLLECTIONS[dataset]
    metadata_types = _metadata_types(dataset, fetcher)

    items = fetcher.items(collection_id)
    if filters.stations is not None:
        items = [item for item in items if item.get("id", "").lower() in filters.stations]
    csv_hrefs = hrefs(
        select(
            assets_of(items, suffixes=(".csv",)),
            time_slices=list(filters.time_slices),
            granularities=filters.granularities,
        )
    )
    return _read_csv_hrefs(dataset, filters, csv_hrefs, fetcher=fetcher, metadata_types=metadata_types)


def read_forecast(dataset: str, filters: Filters, *, fetcher: Fetcher) -> pl.DataFrame:
    """Newest Forecast run, normalized to include its derived Reference timestamp."""
    collection_id = COLLECTIONS[dataset]
    metadata_types = _metadata_types(dataset, fetcher)
    items = fetcher.items(collection_id)
    if filters.stations is not None:
        items = [item for item in items if item.get("id", "").lower() in filters.stations]

    # A forecast item is one day, not one Forecast run, and the newest day can be
    # empty while publication is in progress. Select the newest run by filename.
    items.sort(key=lambda item: item.get("id", ""))
    csv_hrefs = hrefs(select(assets_of(items, suffixes=(".csv",)), latest_run=True))
    return _read_csv_hrefs(dataset, filters, csv_hrefs, fetcher=fetcher, metadata_types=metadata_types)


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
            # Asked of the name, before the member is read: an unwanted station's
            # CSV is never parsed, and the archive's metadata CSV has no station.
            station = indoor_station(name)
            if station is None:
                continue
            if filters.stations is not None and station.lower() not in filters.stations:
                continue
            with zf.open(name) as fh:
                frames.append(parse_indoor_csv(fh.read(), name))

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
    "read_forecast",
    "read_preamble",
    "read_standard",
]
