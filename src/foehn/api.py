"""Public Python API for foehn."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from foehn.client import _retry_session, _validate_href, download_collection, download_metadata
from foehn.collections import (
    COLLECTION_META,
    COLLECTIONS,
    FORECAST_CSV_COLLECTIONS,
    GRIB2_COLLECTIONS,
    NETCDF_COLLECTIONS,
    NO_GRANULARITY_COLLECTIONS,
)
from foehn.convert import _parse_metadata_types, convert_to_parquet, parse_csv_bytes
from foehn.stac import get_collection_items, get_collection_metadata


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
) -> None:
    """Download a single dataset.

    Args:
        dataset: Dataset name (e.g. "smn"). Use list_datasets() to see options.
        data_dir: Root data directory. Defaults to ./data/meteoswiss.
        time_slice: Time slices to download. Defaults to ["recent"].
        since: ISO timestamp for incremental updates.
    """
    if dataset not in COLLECTIONS:
        raise ValueError(f"Unknown dataset: {dataset!r}. Use list_datasets() to see available datasets.")
    if dataset in GRIB2_COLLECTIONS or dataset in NETCDF_COLLECTIONS:
        raise ValueError(f"Dataset {dataset!r} is a binary/grid dataset. Use the CLI with --grids instead.")

    data_dir = Path(data_dir) if data_dir else Path.cwd() / "data" / "meteoswiss"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    download_metadata(dataset, raw_dir)
    download_collection(dataset, raw_dir, data_types=time_slice or ["recent"], since=since)


def to_parquet(
    dataset: str,
    *,
    data_dir: Path | str | None = None,
) -> None:
    """Convert downloaded CSVs to Parquet for a single dataset.

    Args:
        dataset: Dataset name (e.g. "smn").
        data_dir: Root data directory. Defaults to ./data/meteoswiss.
    """
    if dataset not in COLLECTIONS:
        raise ValueError(f"Unknown dataset: {dataset!r}. Use list_datasets() to see available datasets.")

    data_dir = Path(data_dir) if data_dir else Path.cwd() / "data" / "meteoswiss"
    raw_dir = data_dir / "raw"
    parquet_dir = data_dir / "parquet"
    convert_to_parquet(dataset, raw_dir, parquet_dir)


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
    session = _retry_session()

    coll = get_collection_metadata(collection_id)
    for asset_info in coll.get("assets", {}).values():
        href = asset_info.get("href", "")
        if href.endswith(".csv") and suffix in href:
            _validate_href(href)
            resp = session.get(href, timeout=60)
            resp.raise_for_status()
            try:
                content = resp.content.decode("windows-1252")
            except UnicodeDecodeError:
                content = resp.content.decode("utf-8")
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
    lat, lon, data_since.

    Args:
        dataset: Dataset name (e.g. "smn"). Use list_datasets() to see options.
    """
    df = _fetch_metadata_csv(dataset, "_meta_stations")
    return df.select(
        pl.col("station_abbr").alias("abbr"),
        pl.col("station_name").alias("name"),
        pl.col("station_canton").alias("canton"),
        pl.col("station_height_masl").alias("altitude"),
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


def load(
    dataset: str,
    *,
    station: str | list[str] | None = None,
    frequency: str | list[str] | None = None,
    time_slice: str | list[str] | None = None,
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

    Returns:
        A Polars DataFrame containing all matching CSV data.

    Example::

        import foehn

        # Recent daily data for Bern
        df = foehn.load("smn", station="BER", frequency="d")

        # Hourly data for multiple stations
        df = foehn.load("smn", station=["BER", "ZUR"], frequency="h")

        # All recent data (large!)
        df = foehn.load("smn")
    """
    if dataset not in COLLECTIONS:
        raise ValueError(f"Unknown dataset: {dataset!r}. Use list_datasets() to see available datasets.")
    if dataset in GRIB2_COLLECTIONS or dataset in NETCDF_COLLECTIONS:
        raise ValueError(f"Dataset {dataset!r} is a binary/grid dataset and cannot be loaded as a DataFrame.")

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
    session = _retry_session()

    # 1. Fetch metadata types for schema inference.
    metadata_types: dict[str, pl.DataType] = {}
    coll = get_collection_metadata(collection_id)
    for asset_info in coll.get("assets", {}).values():
        href = asset_info.get("href", "")
        if href.endswith(".csv") and "_meta_parameters" in href:
            _validate_href(href)
            resp = session.get(href, timeout=60)
            resp.raise_for_status()
            try:
                content = resp.content.decode("windows-1252")
            except UnicodeDecodeError:
                content = resp.content.decode("utf-8")
            metadata_types = _parse_metadata_types(content)
            break

    # 2. Get STAC items and collect matching CSV URLs.
    items = get_collection_items(collection_id, verbose=False)

    # Filter items by station (item id = station abbreviation).
    if station_filter is not None:
        items = [item for item in items if item.get("id", "").lower() in station_filter]

    if dataset in FORECAST_CSV_COLLECTIONS and items:
        items.sort(
            key=lambda x: x.get("properties", {}).get("datetime", x.get("id", "")),
            reverse=True,
        )
        items = items[:1]

    skip_data_type_filter = dataset in FORECAST_CSV_COLLECTIONS
    csv_hrefs: list[str] = []
    for item in items:
        assets = item.get("assets", {})
        for asset_info in assets.values():
            href = asset_info.get("href", "")
            if not href.endswith(".csv"):
                continue
            filename = href.split("?")[0].split("/")[-1]
            # Filter by frequency — encoded as _{f}_ or _{f}. in the filename.
            if freq_filter is not None:
                parts = filename.rsplit(".", 1)[0].split("_")
                # Frequency is the segment after the station abbr (e.g. ogd-smn_ber_d_recent)
                file_freq = parts[2] if len(parts) > 2 else None
                if file_freq not in freq_filter:
                    continue
            if not skip_data_type_filter:
                has_time_slice = any(ts in href for ts in ("historical", "recent", "now"))
                if has_time_slice and not any(ts in href for ts in time_slice):
                    continue
            csv_hrefs.append(href)

    if not csv_hrefs:
        filters = f"station={station}, frequency={frequency}, time_slice={time_slice}"
        raise ValueError(f"No CSV files found for {dataset!r} with {filters}.")

    # 3. Download and parse each CSV in memory.
    frames: list[pl.DataFrame] = []
    for href in csv_hrefs:
        _validate_href(href)
        resp = session.get(href, timeout=60)
        resp.raise_for_status()
        try:
            content = resp.content.decode("windows-1252")
        except UnicodeDecodeError:
            content = resp.content.decode("utf-8")
        df = parse_csv_bytes(content.encode("utf-8"), metadata_types)
        frames.append(df)

    return pl.concat(frames, how="diagonal_relaxed")
