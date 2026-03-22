"""Public Python API for foehn."""

from __future__ import annotations

from pathlib import Path

from foehn.collections import (
    COLLECTIONS,
    GRIB2_COLLECTIONS,
    NETCDF_COLLECTIONS,
    discover,
)
from foehn.convert import convert_to_parquet
from foehn.download import download_collection, download_metadata


def list_collections() -> list[dict[str, str]]:
    """Return metadata about all available collections.

    Each dict has keys: ``category``, ``key``, ``collection_id``.
    """
    return discover()


def fetch(
    key: str,
    *,
    data_dir: Path | str | None = None,
    data_types: list[str] | None = None,
    since: str | None = None,
) -> None:
    """Download a single collection by key.

    Args:
        key: Collection key (e.g. "smn"). Use list_collections() to see options.
        data_dir: Root data directory. Defaults to ./data/meteoswiss.
        data_types: Time slices to download. Defaults to ["recent"].
        since: ISO timestamp for incremental updates.
    """
    if key not in COLLECTIONS:
        raise ValueError(f"Unknown collection key: {key!r}. Use list_collections() to see available keys.")
    if key in GRIB2_COLLECTIONS or key in NETCDF_COLLECTIONS:
        raise ValueError(f"Collection {key!r} is a binary/grid collection. Use the CLI with --grids instead.")

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
    """Convert downloaded CSVs to Parquet for a single collection.

    Args:
        key: Collection key (e.g. "smn").
        data_dir: Root data directory. Defaults to ./data/meteoswiss.
    """
    if key not in COLLECTIONS:
        raise ValueError(f"Unknown collection key: {key!r}. Use list_collections() to see available keys.")

    data_dir = Path(data_dir) if data_dir else Path.cwd() / "data" / "meteoswiss"
    raw_dir = data_dir / "raw"
    parquet_dir = data_dir / "parquet"
    convert_to_parquet(key, raw_dir, parquet_dir)
