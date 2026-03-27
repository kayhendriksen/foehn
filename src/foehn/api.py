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

    Each dict has keys: ``key``, ``collection_id``, ``category``, ``subcategory``,
    ``description``, ``format``, ``granularities``, ``time_slices``.
    """
    return [{"key": key, "collection_id": cid, **COLLECTION_META[key]} for key, cid in COLLECTIONS.items()]


def download(
    key: str,
    *,
    data_dir: Path | str | None = None,
    data_types: list[str] | None = None,
    since: str | None = None,
) -> None:
    """Download a single dataset by key.

    Args:
        key: Dataset key (e.g. "smn"). Use list_datasets() to see options.
        data_dir: Root data directory. Defaults to ./data/meteoswiss.
        data_types: Time slices to download. Defaults to ["recent"].
        since: ISO timestamp for incremental updates.
    """
    if key not in COLLECTIONS:
        raise ValueError(f"Unknown dataset key: {key!r}. Use list_datasets() to see available keys.")
    if key in GRIB2_COLLECTIONS or key in NETCDF_COLLECTIONS:
        raise ValueError(f"Dataset {key!r} is a binary/grid dataset. Use the CLI with --grids instead.")

    data_dir = Path(data_dir) if data_dir else Path.cwd() / "data" / "meteoswiss"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    download_metadata(key, raw_dir)
    download_collection(key, raw_dir, data_types=data_types or ["recent"], since=since)


def convert(
    key: str,
    *,
    data_dir: Path | str | None = None,
) -> None:
    """Convert downloaded CSVs to Parquet for a single dataset.

    Args:
        key: Dataset key (e.g. "smn").
        data_dir: Root data directory. Defaults to ./data/meteoswiss.
    """
    if key not in COLLECTIONS:
        raise ValueError(f"Unknown dataset key: {key!r}. Use list_datasets() to see available keys.")

    data_dir = Path(data_dir) if data_dir else Path.cwd() / "data" / "meteoswiss"
    raw_dir = data_dir / "raw"
    parquet_dir = data_dir / "parquet"
    convert_to_parquet(key, raw_dir, parquet_dir)


def load(
    key: str,
    *,
    station: str | list[str] | None = None,
    granularity: str | list[str] | None = None,
    data_types: str | list[str] | None = None,
) -> pl.DataFrame:
    """Load a dataset and return it as an in-memory Polars DataFrame.

    No files are written to disk. Data is fetched from the MeteoSwiss STAC API,
    parsed directly in memory, and returned as a single concatenated DataFrame.

    Args:
        key: Dataset key (e.g. "smn"). Use list_datasets() to see options.
        station: Station abbreviation(s) to include (e.g. "BER" or ["BER", "ZUR"]).
            Filters at the STAC item level so unmatched stations are never downloaded.
            Case-insensitive. If None, all stations are included.
        granularity: Time granularity filter(s). Options: "t" (10-min), "h" (hourly),
            "d" (daily), "m" (monthly), "y" (yearly). Can be a single string or list.
            If None, all granularities are included.
        data_types: Time slice(s) to include. Defaults to ["recent"].
            Options: "historical", "recent", "now". Can be a single string or list.

    Returns:
        A Polars DataFrame containing all matching CSV data.

    Example::

        import foehn

        # Recent daily data for Bern
        df = foehn.load("smn", station="BER", granularity="d")

        # Hourly data for multiple stations
        df = foehn.load("smn", station=["BER", "ZUR"], granularity="h")

        # All recent data (large!)
        df = foehn.load("smn")
    """
    if key not in COLLECTIONS:
        raise ValueError(f"Unknown dataset key: {key!r}. Use list_datasets() to see available keys.")
    if key in GRIB2_COLLECTIONS or key in NETCDF_COLLECTIONS:
        raise ValueError(f"Dataset {key!r} is a binary/grid dataset and cannot be loaded as a DataFrame.")

    if data_types is None:
        data_types = ["recent"]
    elif isinstance(data_types, str):
        data_types = [data_types]

    # Normalise station filter to a set of lowercase abbreviations.
    station_filter: set[str] | None = None
    if station is not None:
        if isinstance(station, str):
            station_filter = {station.lower()}
        else:
            station_filter = {s.lower() for s in station}

    # Normalise granularity filter to a set (e.g. {"d", "h"}).
    gran_filter: set[str] | None = None
    if granularity is not None:
        if key in NO_GRANULARITY_COLLECTIONS:
            raise ValueError(f"Collection {key!r} does not support granularity filtering.")
        if isinstance(granularity, str):
            gran_filter = {granularity.lower()}
        else:
            gran_filter = {g.lower() for g in granularity}

    collection_id = COLLECTIONS[key]
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

    if key in FORECAST_CSV_COLLECTIONS and items:
        items.sort(
            key=lambda x: x.get("properties", {}).get("datetime", x.get("id", "")),
            reverse=True,
        )
        items = items[:1]

    skip_data_type_filter = key in FORECAST_CSV_COLLECTIONS
    csv_hrefs: list[str] = []
    for item in items:
        assets = item.get("assets", {})
        for asset_info in assets.values():
            href = asset_info.get("href", "")
            if not href.endswith(".csv"):
                continue
            filename = href.split("?")[0].split("/")[-1]
            # Filter by granularity — encoded as _{g}_ or _{g}. in the filename.
            if gran_filter is not None:
                parts = filename.rsplit(".", 1)[0].split("_")
                # Granularity is the segment after the station abbr (e.g. ogd-smn_ber_d_recent)
                file_gran = parts[2] if len(parts) > 2 else None
                if file_gran not in gran_filter:
                    continue
            if not skip_data_type_filter:
                has_time_slice = any(ts in href for ts in ("historical", "recent", "now"))
                if has_time_slice and not any(dt in href for dt in data_types):
                    continue
            csv_hrefs.append(href)

    if not csv_hrefs:
        filters = f"station={station}, granularity={granularity}, data_types={data_types}"
        raise ValueError(f"No CSV files found for {key!r} with {filters}.")

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
