"""Public Python API for foehn."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

from foehn._urls import asset_filename, clean_href
from foehn.client import (
    DownloadResult,
    _check_zip_size,
    download_climate_scenarios_indoor,
    download_collection,
    download_grib2,
    download_metadata,
    download_netcdf,
)
from foehn.collections import (
    COLLECTION_META,
    COLLECTIONS,
    CSV_ZIP_COLLECTIONS,
    FORECAST_CSV_COLLECTIONS,
    GRIB2_COLLECTIONS,
    NETCDF_COLLECTIONS,
    NO_GRANULARITY_COLLECTIONS,
    PREAMBLE_CSV_COLLECTIONS,
    forecast_run_from_filename,
    time_slice_from_filename,
)
from foehn.convert import (
    _parse_metadata_types,
    add_forecast_local_timestamp,
    add_indoor_columns,
    convert_climate_scenarios_indoor_to_parquet,
    convert_climate_scenarios_to_parquet,
    convert_to_parquet,
    decode_meteoswiss_csv,
    parse_climate_scenarios_csv,
    parse_csv_bytes,
    parse_indoor_filename,
    utf8_meteoswiss_csv,
)
from foehn.fetch import DEFAULT_WORKERS, default_fetcher
from foehn.grids import open_dataset, to_zarr

__all__ = [
    "download",
    "inventory",
    "list_datasets",
    "load",
    "open_dataset",
    "parameters",
    "stations",
    "to_parquet",
    "to_zarr",
]


def list_datasets() -> list[dict]:
    """Return metadata about all available datasets.

    Each dict has keys: ``dataset``, ``collection_id``, ``category``, ``subcategory``,
    ``description``, ``format``, ``frequencies``, ``time_slices``.
    """
    return [{"dataset": key, "collection_id": cid, **COLLECTION_META[key]} for key, cid in COLLECTIONS.items()]


def download(
    dataset: str,
    *,
    data_dir: Path | str | None = None,
    time_slice: list[str] | None = None,
    since: str | None = None,
    workers: int = DEFAULT_WORKERS,
    force: bool = False,
) -> DownloadResult:
    """Download a single dataset.

    Args:
        dataset: Dataset name (e.g. "smn"). Use list_datasets() to see options.
        data_dir: Root data directory. Defaults to ./data/meteoswiss.
        time_slice: Time slices to download. Defaults to ["recent"]. Ignored for
            binary/grid datasets (GRIB2/NetCDF), which fetch the latest assets.
        since: ISO timestamp for incremental updates — only assets updated after
            it are fetched. This function does not track state itself: pass the
            previous run's timestamp, or use the ``foehn download`` CLI, which
            persists one in ``_last_run.json`` and only advances it when the run
            fully succeeds.
        workers: Concurrent HTTP downloads (default 8).
        force: Re-download even when local files look up to date. Currently
            only affects ZIP-shipped datasets (e.g. climate_scenarios_indoor),
            which otherwise skip when already extracted; other formats refresh
            via ``since``/ETags.

    Returns:
        DownloadResult summarising the download. Use ``result.downloaded > 0``
        to gate expensive downstream processing and ``result.failed`` to detect
        partial failures.
    """
    if dataset not in COLLECTIONS:
        raise ValueError(f"Unknown dataset: {dataset!r}. Use list_datasets() to see available datasets.")

    data_dir = Path(data_dir) if data_dir else Path.cwd() / "data" / "meteoswiss"
    bronze_dir = data_dir / "bronze"
    bronze_dir.mkdir(parents=True, exist_ok=True)
    fetcher = default_fetcher()

    # Binary/grid datasets have no Parquet path — download the raw assets so the
    # Python API mirrors the CLI's --grids behaviour (and open_dataset/to_zarr).
    if dataset in GRIB2_COLLECTIONS:
        return download_grib2(dataset, bronze_dir, since=since, workers=workers, fetcher=fetcher)
    if dataset in NETCDF_COLLECTIONS:
        return download_netcdf(dataset, bronze_dir, since=since, workers=workers, fetcher=fetcher)

    if dataset in CSV_ZIP_COLLECTIONS:
        return download_climate_scenarios_indoor(bronze_dir, dataset, force=force, fetcher=fetcher)

    meta = download_metadata(dataset, bronze_dir, workers=workers, fetcher=fetcher)
    coll = download_collection(
        dataset, bronze_dir, data_types=time_slice or ["recent"], since=since, workers=workers, fetcher=fetcher
    )

    return DownloadResult(
        total_assets=meta.total_assets + coll.total_assets,
        downloaded=meta.downloaded + coll.downloaded,
        skipped=meta.skipped + coll.skipped,
        failed=meta.failed + coll.failed,
        filenames=meta.filenames + coll.filenames,
    )


def to_parquet(
    dataset: str,
    *,
    data_dir: Path | str | None = None,
) -> None:
    """Convert downloaded CSVs to Parquet for a single dataset.

    Args:
        dataset: Dataset name (e.g. "smn").
        data_dir: Root data directory. Defaults to ./data/meteoswiss.

    Raises:
        RuntimeError: If one or more groups fail to convert. Each failure is
            also printed to stdout. Mirrors the CLI's exit-1 semantics so
            Python callers can't accidentally treat partial output as success.
    """
    if dataset not in COLLECTIONS:
        raise ValueError(f"Unknown dataset: {dataset!r}. Use list_datasets() to see available datasets.")

    data_dir = Path(data_dir) if data_dir else Path.cwd() / "data" / "meteoswiss"
    bronze_dir = data_dir / "bronze"
    parquet_dir = data_dir / "parquet"
    if dataset in CSV_ZIP_COLLECTIONS:
        failures = convert_climate_scenarios_indoor_to_parquet(bronze_dir, parquet_dir)
    elif dataset in PREAMBLE_CSV_COLLECTIONS:
        failures = convert_climate_scenarios_to_parquet(bronze_dir, parquet_dir)
    else:
        failures = convert_to_parquet(dataset, bronze_dir, parquet_dir)
    if failures:
        raise RuntimeError(
            f"to_parquet({dataset!r}) failed: {failures} group(s) did not convert. See stdout for details."
        )


def _fetch_metadata_csv(dataset: str, suffix: str) -> pl.DataFrame:
    """Fetch a collection-level metadata CSV from the STAC API.

    Args:
        dataset: Dataset name (e.g. "smn").
        suffix: Metadata file suffix (e.g. "_meta_parameters").

    Returns:
        A Polars DataFrame with the parsed CSV contents.
    """
    if dataset not in COLLECTIONS:
        raise ValueError(f"Unknown dataset: {dataset!r}. Use list_datasets() to see available datasets.")

    collection_id = COLLECTIONS[dataset]
    fetcher = default_fetcher()

    coll = fetcher.collection(collection_id)
    for asset_info in coll.get("assets", {}).values():
        href = asset_info.get("href", "")
        if clean_href(href).endswith(".csv") and suffix in href:
            content = decode_meteoswiss_csv(fetcher.get(href, timeout=60).body)
            return pl.read_csv(content.encode("utf-8"), separator=";")

    raise ValueError(f"No {suffix} metadata found for dataset {dataset!r}.")


def parameters(dataset: str) -> pl.DataFrame:
    """Fetch parameter metadata for a dataset.

    Returns a DataFrame with columns: shortname, description, unit, type,
    granularity, decimals, group.

    Args:
        dataset: Dataset name (e.g. "smn"). Use list_datasets() to see options.
    """
    df = _fetch_metadata_csv(dataset, "_meta_parameters")
    return df.select(
        pl.col("parameter_shortname").alias("shortname"),
        pl.col("parameter_description_en").alias("description"),
        pl.col("parameter_unit").alias("unit"),
        pl.col("parameter_datatype").alias("type"),
        pl.col("parameter_granularity").alias("granularity"),
        pl.col("parameter_decimals").alias("decimals"),
        pl.col("parameter_group_en").alias("group"),
    )


def stations(dataset: str) -> pl.DataFrame:
    """Fetch station metadata for a dataset.

    Returns a DataFrame with columns: abbr, name, canton, altitude,
    lv95_east, lv95_north, lat, lon, data_since. ``data_since`` keeps the
    MeteoSwiss station metadata format (DD.MM.YYYY).

    Args:
        dataset: Dataset name (e.g. "smn"). Use list_datasets() to see options.
    """
    df = _fetch_metadata_csv(dataset, "_meta_stations")
    return df.select(
        pl.col("station_abbr").alias("abbr"),
        pl.col("station_name").alias("name"),
        pl.col("station_canton").alias("canton"),
        pl.col("station_height_masl").alias("altitude"),
        pl.col("station_coordinates_lv95_east").alias("lv95_east"),
        pl.col("station_coordinates_lv95_north").alias("lv95_north"),
        pl.col("station_coordinates_wgs84_lat").alias("lat"),
        pl.col("station_coordinates_wgs84_lon").alias("lon"),
        pl.col("station_data_since").alias("data_since"),
    )


def inventory(dataset: str) -> pl.DataFrame:
    """Fetch the data inventory for a dataset.

    Returns a DataFrame with columns: station, parameter, data_since,
    data_till, owner.

    Args:
        dataset: Dataset name (e.g. "smn"). Use list_datasets() to see options.
    """
    df = _fetch_metadata_csv(dataset, "_meta_datainventory")
    return df.select(
        pl.col("station_abbr").alias("station"),
        pl.col("parameter_shortname").alias("parameter"),
        pl.col("data_since"),
        pl.col("data_till"),
        pl.col("owner"),
    )


# A ``date_to`` of exactly "YYYY-MM-DD" names a whole day, not its midnight.
_BARE_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


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


def _apply_time_filters(
    df: pl.DataFrame,
    *,
    year: int | list[int] | None = None,
    month: int | list[int] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> pl.DataFrame:
    """Apply the timestamp row predicates shared by the post-filter and the per-frame pass.

    Split out of :func:`_apply_post_filters` so ``load()`` can run it on each CSV
    as it is parsed, instead of only on the concatenated result. These are all
    per-row predicates, so filtering early is equivalent — but it bounds peak
    memory by the largest single station file rather than by the whole matched
    set. ``drop_null`` deliberately stays in the post-filter: a frame missing
    that column keeps every row here, while after a diagonal concat the column
    exists as null across those rows and they are dropped.
    """
    ts = "reference_timestamp"
    if year is not None:
        years = [year] if isinstance(year, int) else year
        df = df.filter(pl.col(ts).dt.year().is_in(years))
    if month is not None:
        months = [month] if isinstance(month, int) else month
        df = df.filter(pl.col(ts).dt.month().is_in(months))
    # Cast the timestamp column to Datetime before comparing: some daily/monthly
    # files parse ``reference_timestamp`` as a Date, and comparing Date vs the
    # Datetime literal would raise. Date→Datetime and Datetime→Datetime are both safe.
    if date_from is not None:
        df = df.filter(pl.col(ts).cast(pl.Datetime) >= pl.lit(date_from).str.to_datetime())
    if date_to is not None:
        bound = pl.lit(date_to).str.to_datetime()
        if _BARE_DATE_RE.match(date_to):
            # A bare "YYYY-MM-DD" means the whole of that day. Comparing <= the
            # parsed midnight is right for d/m/y (timestamps sit at 00:00) but
            # silently drops every 10-minute and hourly reading after 00:00, so
            # bound the day exclusively at the next midnight instead.
            df = df.filter(pl.col(ts).cast(pl.Datetime) < bound.dt.offset_by("1d"))
        else:
            df = df.filter(pl.col(ts).cast(pl.Datetime) <= bound)
    return df


def _apply_post_filters(
    df: pl.DataFrame,
    *,
    year: int | list[int] | None = None,
    month: int | list[int] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    drop_null: str | None = None,
    sort: str | None = None,
    columns: list[str] | None = None,
    limit: int | None = None,
    keep_cols: tuple[str, ...] = ("station_abbr", "reference_timestamp"),
) -> pl.DataFrame:
    """Apply the shared in-memory row/column filters used by load() variants."""
    ts = "reference_timestamp"
    df = _apply_time_filters(df, year=year, month=month, date_from=date_from, date_to=date_to)
    if drop_null:
        _require_columns(df, [drop_null], "drop_null")
        df = df.filter(pl.col(drop_null).is_not_null())
    if sort in ("asc", "desc"):
        df = df.sort(ts, descending=(sort == "desc"))
    if columns:
        _require_columns(df, columns, "columns")
        keep = [c for c in keep_cols if c in df.columns]
        keep += [c for c in columns if c not in keep]
        df = df.select(keep)
    if limit is not None:
        df = df.head(limit)
    return df


def _load_indoor(
    dataset: str,
    *,
    station: str | list[str] | None = None,
    year: int | list[int] | None = None,
    month: int | list[int] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    columns: list[str] | None = None,
    drop_null: str | None = None,
    sort: str | None = None,
    limit: int | None = None,
) -> pl.DataFrame:
    """Load a zipped multi-CSV collection (indoor scenarios) into a DataFrame.

    Unlike the per-station collections, this is a single archive, so the whole
    ZIP is fetched and parsed in memory; ``station`` filters which member CSVs
    are parsed, the rest of the filters apply to the combined frame.
    """
    import io
    import zipfile

    collection_id = COLLECTIONS[dataset]
    fetcher = default_fetcher()
    items = fetcher.items(collection_id)
    zip_href = next(
        (
            asset_info.get("href", "")
            for item in items
            for asset_info in item.get("assets", {}).values()
            if clean_href(asset_info.get("href", "")).endswith(".zip")
        ),
        None,
    )
    if not zip_href:
        raise ValueError(f"No .zip asset found for {dataset!r}.")

    station_filter: set[str] | None = None
    if station is not None:
        station_filter = {station.lower()} if isinstance(station, str) else {s.lower() for s in station}

    archive = fetcher.get(zip_href, timeout=300).body

    frames: list[pl.DataFrame] = []
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        # Everything below is parsed in memory — refuse a decompression bomb.
        _check_zip_size(zf, zip_href.split("/")[-1])
        for name in zf.namelist():
            if not name.endswith(".csv"):
                continue
            parsed = parse_indoor_filename(Path(name).stem)
            if parsed is None:
                continue
            st, period, scenario, variant = parsed
            if station_filter is not None and st.lower() not in station_filter:
                continue
            with zf.open(name) as fh:
                frame = pl.read_csv(fh.read(), separator=",", infer_schema_length=10_000, truncate_ragged_lines=True)
            frames.append(add_indoor_columns(frame, st, period, scenario, variant))

    if not frames:
        raise ValueError(f"No indoor data found for {dataset!r} with station={station}.")

    df = pl.concat(frames, how="diagonal_relaxed")
    return _apply_post_filters(
        df,
        year=year,
        month=month,
        date_from=date_from,
        date_to=date_to,
        drop_null=drop_null,
        sort=sort,
        columns=columns,
        limit=limit,
        keep_cols=("station_abbr", "reference_timestamp", "period", "scenario", "variant"),
    )


def _load_climate_scenarios(
    dataset: str,
    *,
    station: str | list[str] | None = None,
    columns: list[str] | None = None,
    drop_null: str | None = None,
    sort: str | None = None,
    limit: int | None = None,
    workers: int = DEFAULT_WORKERS,
) -> pl.DataFrame:
    """Load CH2025 climate-scenario CSVs (metadata preamble + wide model table).

    Dates are nominal (0001..0030 on a 365-day calendar), so the calendar-based
    year/month/date filters do not apply here; ``sort`` orders lexically by the
    string ``date`` column.
    """
    collection_id = COLLECTIONS[dataset]
    fetcher = default_fetcher()

    station_filter: set[str] | None = None
    if station is not None:
        station_filter = {station.lower()} if isinstance(station, str) else {s.lower() for s in station}

    items = fetcher.items(collection_id)
    if station_filter is not None:
        items = [item for item in items if item.get("id", "").lower() in station_filter]

    hrefs: list[str] = []
    for item in items:
        for asset_info in item.get("assets", {}).values():
            href = asset_info.get("href", "")
            filename = asset_filename(href)
            if filename.endswith(".csv") and "_meta_" not in filename:
                hrefs.append(href)

    if not hrefs:
        raise ValueError(f"No climate-scenario CSVs found for {dataset!r} with station={station}.")

    # The fetcher is safe to share across the pool: it hands each worker thread
    # its own session.
    def _fetch(href: str) -> pl.DataFrame:
        return parse_climate_scenarios_csv(fetcher.get(href, timeout=120).body, asset_filename(href))

    if len(hrefs) == 1 or workers <= 1:
        frames = [_fetch(h) for h in hrefs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            frames = list(pool.map(_fetch, hrefs))

    df = pl.concat(frames, how="diagonal_relaxed")

    if drop_null:
        _require_columns(df, [drop_null], "drop_null")
        df = df.filter(pl.col(drop_null).is_not_null())
    if sort in ("asc", "desc"):
        df = df.sort("date", descending=(sort == "desc"))
    if columns:
        _require_columns(df, columns, "columns")
        keep = [c for c in ("station_abbr", "variable", "gwl", "date") if c in df.columns]
        keep += [c for c in columns if c not in keep]
        df = df.select(keep)
    if limit is not None:
        df = df.head(limit)
    return df


def load(
    dataset: str,
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
) -> pl.DataFrame:
    """Load a dataset and return it as an in-memory Polars DataFrame.

    No files are written to disk. Data is fetched from the MeteoSwiss STAC API,
    parsed directly in memory, and returned as a single concatenated DataFrame.

    Args:
        dataset: Dataset name (e.g. "smn"). Use list_datasets() to see options.
        station: Station abbreviation(s) to include (e.g. "BER" or ["BER", "ZUR"]).
            Filters at the STAC item level so unmatched stations are never downloaded.
            Case-insensitive. If None, all stations are included.
        frequency: Time frequency filter(s). Options: "t" (10-min), "h" (hourly),
            "d" (daily), "m" (monthly), "y" (yearly). Can be a single string or list.
            If None, all frequencies are included.
        time_slice: Time slice(s) to include. Defaults to ["recent"].
            Options: "historical", "recent", "now". Can be a single string or list.
        year: Filter to specific year(s) (e.g. 2025 or [2020, 2021, 2022]).
        month: Filter to specific month(s) (1-12, e.g. 7 or [6, 7, 8] for summer).
        date_from: Start date (inclusive), ISO format "YYYY-MM-DD".
        date_to: End date (inclusive), ISO format "YYYY-MM-DD".
        columns: Only return these columns. ``station_abbr`` and
            ``reference_timestamp`` are always included.
        drop_null: Drop rows where this column is null.
        sort: Sort by timestamp. Options: "asc" (oldest first) or "desc"
            (newest first).
        limit: Cap the returned DataFrame to N rows. Applied after sort/columns.
            NOTE: this only bounds the returned shape — it does not reduce
            network bytes, since CSVs are per-station and not pre-bucketed by
            row count. To reduce wire volume, narrow with ``time_slice="now"``,
            ``year``, or ``date_from/date_to``.
        workers: Concurrent CSV downloads (default 8). The CSV fetches are
            parallelised via a ThreadPoolExecutor; metadata fetch stays serial.

    Returns:
        A Polars DataFrame containing all matching CSV data.

    Example::

        import foehn

        # Recent daily data for Bern
        df = foehn.load("smn", station="BER", frequency="d")

        # Hourly data for multiple stations
        df = foehn.load("smn", station=["BER", "ZUR"], frequency="h")

        # Daily data for Bern, only January 2026
        df = foehn.load("smn", station="BER", frequency="d", year=2026, month=1)

        # Latest 10 readings (sort+limit are applied after the CSV is parsed)
        df = foehn.load("smn", station="BER", frequency="t",
                        time_slice="now", sort="desc", limit=10)

        # Summer 2025 temperatures, sorted newest first
        df = foehn.load("smn", station="BER", frequency="d",
                        time_slice="historical", date_from="2025-06-01",
                        date_to="2025-08-31", columns=["tre200d0"],
                        sort="desc")
    """
    if dataset not in COLLECTIONS:
        raise ValueError(f"Unknown dataset: {dataset!r}. Use list_datasets() to see available datasets.")
    if dataset in GRIB2_COLLECTIONS or dataset in NETCDF_COLLECTIONS:
        raise ValueError(f"Dataset {dataset!r} is a binary/grid dataset and cannot be loaded as a DataFrame.")
    if dataset in CSV_ZIP_COLLECTIONS:
        if frequency is not None:
            raise ValueError(f"Dataset {dataset!r} does not support frequency filtering.")
        return _load_indoor(
            dataset,
            station=station,
            year=year,
            month=month,
            date_from=date_from,
            date_to=date_to,
            columns=columns,
            drop_null=drop_null,
            sort=sort,
            limit=limit,
        )

    if dataset in PREAMBLE_CSV_COLLECTIONS:
        if frequency is not None:
            raise ValueError(f"Dataset {dataset!r} does not support frequency filtering.")
        if any(x is not None for x in (year, month, date_from, date_to)):
            raise ValueError(
                f"Dataset {dataset!r} uses nominal 30-year dates (0001..0030); "
                "year/month/date_from/date_to filters are not supported."
            )
        return _load_climate_scenarios(
            dataset,
            station=station,
            columns=columns,
            drop_null=drop_null,
            sort=sort,
            limit=limit,
            workers=workers,
        )

    if time_slice is None:
        time_slice = ["recent"]
    elif isinstance(time_slice, str):
        time_slice = [time_slice]

    # Normalise station filter to a set of lowercase abbreviations.
    station_filter: set[str] | None = None
    if station is not None:
        if isinstance(station, str):
            station_filter = {station.lower()}
        else:
            station_filter = {s.lower() for s in station}

    # Normalise frequency filter to a set (e.g. {"d", "h"}).
    freq_filter: set[str] | None = None
    if frequency is not None:
        if dataset in NO_GRANULARITY_COLLECTIONS:
            raise ValueError(f"Dataset {dataset!r} does not support frequency filtering.")
        if isinstance(frequency, str):
            freq_filter = {frequency.lower()}
        else:
            freq_filter = {f.lower() for f in frequency}

    collection_id = COLLECTIONS[dataset]
    fetcher = default_fetcher()

    # 1. Fetch metadata types for schema inference.
    metadata_types: dict[str, type[pl.DataType]] = {}
    coll = fetcher.collection(collection_id)
    for asset_info in coll.get("assets", {}).values():
        href = asset_info.get("href", "")
        if clean_href(href).endswith(".csv") and "_meta_parameters" in href:
            metadata_types = _parse_metadata_types(decode_meteoswiss_csv(fetcher.get(href, timeout=60).body))
            break

    # 2. Get STAC items and collect matching CSV URLs.
    items = fetcher.items(collection_id)

    # Filter items by station (item id = station abbreviation).
    if station_filter is not None:
        items = [item for item in items if item.get("id", "").lower() in station_filter]

    # A forecast item is one *day*, not one forecast, and the newest one is empty
    # until that day's runs publish — so keep all items and narrow to the newest
    # run by filename below. Ranking on ``datetime`` would not help either: it is a
    # refresh timestamp, identical across items to the microsecond.
    if dataset in FORECAST_CSV_COLLECTIONS and items:
        items.sort(key=lambda x: x.get("id", ""))

    skip_data_type_filter = dataset in FORECAST_CSV_COLLECTIONS
    csv_hrefs: list[str] = []
    for item in items:
        assets = item.get("assets", {})
        for asset_info in assets.values():
            href = asset_info.get("href", "")
            filename = asset_filename(href)
            if not filename.endswith(".csv"):
                continue
            # Filter by frequency — encoded as _{f}_ or _{f}. in the filename.
            if freq_filter is not None:
                parts = filename.rsplit(".", 1)[0].split("_")
                # Frequency is the segment after the station abbr (e.g. ogd-smn_ber_d_recent)
                file_freq = parts[2] if len(parts) > 2 else None
                if file_freq not in freq_filter:
                    continue
            if not skip_data_type_filter:
                slice_ = time_slice_from_filename(filename)
                if slice_ is not None and slice_ not in time_slice:
                    continue
            csv_hrefs.append(href)

    # Narrow forecasts to the newest model run — the retained window holds ~40 runs
    # of ~32 files each, and load() wants the current forecast, not all of them.
    if dataset in FORECAST_CSV_COLLECTIONS and csv_hrefs:
        runs = {run for href in csv_hrefs if (run := forecast_run_from_filename(clean_href(href))) is not None}
        if runs:
            latest_run = max(runs)
            csv_hrefs = [href for href in csv_hrefs if forecast_run_from_filename(clean_href(href)) == latest_run]

    if not csv_hrefs:
        filters = f"station={station}, frequency={frequency}, time_slice={time_slice}"
        raise ValueError(f"No CSV files found for {dataset!r} with {filters}.")

    # 3. Download and parse each CSV concurrently. The fetcher is safe to share
    # across the pool: it hands each worker thread its own session.
    # With an explicit ``columns=``, tell the parser up front instead of parsing all
    # ~42 columns of every station file and selecting afterwards. The frame retained
    # per station drops by an order of magnitude, which is what bounds peak memory
    # while the whole matched set is assembled. Everything the later filters and the
    # concat rely on has to survive the projection.
    wanted_columns: set[str] | None = None
    if columns:
        wanted_columns = {"station_abbr", "reference_timestamp", *columns}
        if drop_null:
            wanted_columns.add(drop_null)
        if dataset == "forecast_local":
            wanted_columns.add("Date")  # reference_timestamp is derived from it

    def _fetch(href: str) -> pl.DataFrame:
        # Zero-copy when the payload is already UTF-8 (the usual case): these are
        # the big files, and ``workers`` of them are in flight at once.
        body = fetcher.get(href, timeout=60).body
        frame = parse_csv_bytes(utf8_meteoswiss_csv(body), metadata_types, wanted_columns=wanted_columns)
        # Drop the rows this call can never return *before* they reach the
        # concat. Every frame is otherwise held in full until the whole matched
        # set is materialised, so a narrow year= over many stations peaked at the
        # size of the entire time slice. forecast_local has no reference_timestamp
        # of its own — derive it here so its frames can be narrowed too.
        if dataset == "forecast_local":
            frame = add_forecast_local_timestamp(frame)
        if "reference_timestamp" in frame.columns:
            frame = _apply_time_filters(frame, year=year, month=month, date_from=date_from, date_to=date_to)
        return frame

    if len(csv_hrefs) == 1 or workers <= 1:
        frames = [_fetch(href) for href in csv_hrefs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            frames = list(pool.map(_fetch, csv_hrefs))

    df = pl.concat(frames, how="diagonal_relaxed")

    # forecast_local has no reference_timestamp; derive it from the compact Date column.
    if dataset == "forecast_local":
        df = add_forecast_local_timestamp(df)

    return _apply_post_filters(
        df,
        year=year,
        month=month,
        date_from=date_from,
        date_to=date_to,
        drop_null=drop_null,
        sort=sort,
        columns=columns,
        limit=limit,
    )
