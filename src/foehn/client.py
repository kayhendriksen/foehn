"""Download MeteoSwiss data from the STAC API and opendata.swiss."""

from __future__ import annotations

import json
import logging
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from foehn.assets import Asset, assets_of, collection_assets, latest_run_of, select
from foehn.collections import (
    CLIMATE_NORMALS_ZIP_URL,
    COLLECTIONS,
    DatasetKind,
    kind,
)
from foehn.fetch import DEFAULT_WORKERS, Fetcher
from foehn.transfer import (
    DownloadResult,
    atomic_write_text,
    csv_to_disk,
    fetch_all,
    stream_to_disk,
)

logger = logging.getLogger(__name__)


# --- State files (ETags + last-run timestamp) ---


def load_etags(data_dir: Path) -> dict:
    path = data_dir / "_etags.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read %s (%s) — treating as empty", path, exc)
    return {}


def save_etags(data_dir: Path, etags: dict):
    path = data_dir / "_etags.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(etags, indent=2))


def load_last_run(data_dir: Path) -> str | None:
    """Return ISO timestamp of last successful run, or None."""
    path = data_dir / "_last_run.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read %s (%s) — treating as no previous run", path, exc)
            return None
        return data.get("timestamp")
    return None


def save_last_run(data_dir: Path):
    path = data_dir / "_last_run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps({"timestamp": datetime.now(UTC).isoformat()}))


# --- CSV downloads ---


def download_collection(
    collection_key: str,
    output_dir: Path,
    data_types: list[str] | None = None,
    since: str | None = None,
    workers: int = DEFAULT_WORKERS,
    state_dir: Path | None = None,
    *,
    fetcher: Fetcher,
) -> DownloadResult:
    """Download CSVs for a collection.

    Args:
        collection_key: Key from COLLECTIONS (e.g. "smn").
        output_dir: Root directory for bronze downloads (files go to output_dir/<key>/).
        data_types: List of "historical", "recent", "now". Defaults to ["recent"].
        since: ISO timestamp — only process items updated after this time.
        workers: Number of concurrent HTTP downloads.
        state_dir: Where ``_etags.json`` lives. Defaults to ``output_dir.parent``,
            which is the data root when output_dir is the usual bronze directory.
            Pass it explicitly to keep the ETag state somewhere else — calling
            this with an arbitrary output_dir otherwise scatters state a level up.

    Returns:
        DownloadResult with counts and list of newly downloaded filenames.
    """
    state_dir = state_dir if state_dir is not None else output_dir.parent
    if data_types is None:
        data_types = ["recent"]

    collection_id = COLLECTIONS[collection_key]
    out_dir = output_dir / collection_key

    etags = load_etags(state_dir)

    logger.info("%s", "=" * 60)
    logger.info("Collection: %s", collection_id)
    logger.info("Data types: %s", data_types)
    logger.info("Output dir: %s", out_dir)
    logger.info("%s", "=" * 60)

    items = fetcher.items(collection_id)
    logger.info("  Found %d items", len(items))

    # Filter to items updated since last run
    if since:
        items = [item for item in items if item.get("properties", {}).get("updated", "") > since]
        logger.info("  %d items updated since last run", len(items))
        if not items:
            logger.info("  Nothing changed — skipping")
            return DownloadResult()

    # NB: a forecast item is one *day* (e.g. "20260722-ch") holding that day's ~24
    # hourly runs × 32 parameters, not a single forecast. Selecting the latest item
    # therefore did not select the latest run — and since the newest item is created
    # at ~04:00 UTC and filled as the day's runs publish, it is routinely empty,
    # which is what produced zero CSVs. Every item is kept, and the newest run is
    # picked out of the filenames below. (Ordering the items buys nothing: the run
    # is chosen with max(), and downloads complete out of order regardless.)

    # ``every_csv`` is the full pre-filter set — the prune universe below, so
    # ETags for slices outside this run's data_types are kept.
    is_forecast = kind(collection_key) is DatasetKind.FORECAST_CSV
    every_csv = assets_of(items, suffixes=(".csv",))
    # One forecast run is ~32 files at ~30 MB each (~1 GB); the full retained
    # window is ~40 runs (~40 GB). Forecast filenames carry no time slice, so
    # narrowing to the newest run is what bounds them instead.
    csv_assets = select(
        every_csv,
        time_slices=None if is_forecast else data_types,
        latest_run=is_forecast,
    )
    if is_forecast and (run := latest_run_of(every_csv)) is not None:
        available = len({a.forecast_run for a in every_csv if a.forecast_run is not None})
        logger.info("  Latest forecast run: %s (of %d available)", run, available)

    # The conditional-GET writer decides "unchanged" from the server's 304, so
    # there is no local skip rule here — the ETag store is the skip rule.
    result = fetch_all(
        csv_assets,
        out_dir,
        fetcher=fetcher,
        workers=workers,
        write=csv_to_disk,
        etags=etags,
        label="CSV",
    )

    # Prune ETags for assets that no longer exist upstream, so _etags.json
    # doesn't grow forever (e.g. forecast runs get fresh filenames each cycle).
    # Only on a clean full listing: with ``since`` the item list is partial, and
    # after failures the universe may be incomplete.
    if since is None and result.failed == 0:
        prefix = f"/{collection_id}/"
        listed = {a.href for a in every_csv}
        stale = [k for k in etags if prefix in k and k not in listed]
        for k in stale:
            del etags[k]
        if stale:
            logger.info("  Pruned %d stale ETag entries", len(stale))

    save_etags(state_dir, etags)
    return result


# --- Metadata downloads ---


def download_metadata(
    collection_key: str, output_dir: Path, workers: int = DEFAULT_WORKERS, *, fetcher: Fetcher
) -> DownloadResult:
    """Download collection-level metadata files (stations, parameters, inventory)."""
    coll = fetcher.collection(COLLECTIONS[collection_key])
    return fetch_all(
        collection_assets(coll, suffixes=(".csv",)),
        output_dir / collection_key,
        fetcher=fetcher,
        workers=workers,
        write=csv_to_disk,
        label="metadata file",
    )


# --- GRIB2 / HDF5 downloads ---


def _already_current(asset: Asset, filepath: Path) -> bool:
    """A :data:`~foehn.transfer.SkipRule`: is the local copy of *asset* up to date?

    Handles MeteoSwiss's in-place overwrites — e.g. CombiPrecip reanalysis
    (CPCH) replaces the original CPC hourly file with the same filename
    ~8 days later. A plain exists() check would leave the stale version
    on disk; comparing the STAC "updated" timestamp against local mtime
    picks up those server-side updates.
    """
    if not filepath.exists():
        return False
    if not asset.updated:
        return True
    try:
        remote_dt = datetime.fromisoformat(asset.updated)
        local_dt = datetime.fromtimestamp(filepath.stat().st_mtime, tz=UTC)
    except (ValueError, OSError):
        return True
    return remote_dt <= local_dt


def _exists(_asset: Asset, filepath: Path) -> bool:
    """A :data:`~foehn.transfer.SkipRule` for static assets: fetch each one once."""
    return filepath.exists()


def download_grib2(
    collection_key: str,
    output_dir: Path,
    since: str | None = None,
    workers: int = DEFAULT_WORKERS,
    *,
    fetcher: Fetcher,
) -> DownloadResult:
    """Download GRIB2/HDF5 binary files (latest page only)."""
    collection_id = COLLECTIONS[collection_key]

    logger.info("%s", "=" * 60)
    logger.info("GRIB2 Collection: %s", collection_id)
    logger.info("Output dir: %s", output_dir / collection_key)
    logger.info("%s", "=" * 60)

    # Only the newest page — forecast/radar data is ephemeral and these
    # collections hold thousands of items.
    items = fetcher.items(collection_id, max_items=100)
    logger.info("  Found %d items (latest page)", len(items))

    if since:
        items = [item for item in items if item.get("properties", {}).get("updated", "") > since]
        logger.info("  %d items updated since last run", len(items))
        if not items:
            logger.info("  Nothing changed — skipping")
            return DownloadResult()

    return fetch_all(
        assets_of(items, suffixes=(".grib2", ".h5", ".hdf5")),
        output_dir / collection_key,
        fetcher=fetcher,
        workers=workers,
        write=stream_to_disk,
        skip=_already_current,
        label="binary file",
    )


# --- NetCDF / GeoTIFF / ZIP downloads ---


def download_netcdf(
    collection_key: str,
    output_dir: Path,
    since: str | None = None,
    workers: int = DEFAULT_WORKERS,
    *,
    fetcher: Fetcher,
) -> DownloadResult:
    """Download NetCDF, GeoTIFF, and ZIP files for spatial/static collections.

    Args:
        collection_key: Key from COLLECTIONS.
        output_dir: Root directory for bronze downloads.
        since: ISO timestamp — only process items updated after this time.
        workers: Number of concurrent HTTP downloads.
    """
    collection_id = COLLECTIONS[collection_key]

    logger.info("%s", "=" * 60)
    logger.info("NetCDF Collection: %s", collection_id)
    logger.info("Output dir: %s", output_dir / collection_key)
    logger.info("%s", "=" * 60)

    items = fetcher.items(collection_id)
    logger.info("  Found %d items", len(items))

    if since:
        items = [item for item in items if item.get("properties", {}).get("updated", "") > since]
        logger.info("  %d items updated since last run", len(items))
        if not items:
            logger.info("  Nothing changed — skipping")
            return DownloadResult()

    # These are static: an existing file is never restated upstream, so a plain
    # existence check is enough (unlike the ephemeral GRIB2/radar collections,
    # which MeteoSwiss overwrites in place).
    return fetch_all(
        assets_of(items, suffixes=(".nc", ".tif", ".zip")),
        output_dir / collection_key,
        fetcher=fetcher,
        workers=workers,
        write=stream_to_disk,
        skip=_exists,
        label="file",
    )


# --- ZIP safety (zip-slip + decompression-bomb guards) ---

# Cap on declared total decompressed size. Generous — the largest legitimate
# archive (indoor scenarios) is well under this — but stops a decompression
# bomb from a compromised upstream filling the disk (or RAM, for in-memory
# reads). Python's zipfile enforces each member's declared size on read, so
# checking the headers is sufficient.
_MAX_ZIP_EXTRACT_BYTES = 10 * 1024**3  # 10 GiB


def check_zip_size(zf: zipfile.ZipFile, source: str) -> None:
    """Raise ValueError if the archive declares more decompressed bytes than the cap.

    Public because the archive *load* path reads its ZIP in memory rather than
    through :func:`_safe_extract_zip`, and has to apply the same cap. One rule,
    one place — the reader crossing this seam is part of the interface.
    """
    total = sum(m.file_size for m in zf.infolist())
    if total > _MAX_ZIP_EXTRACT_BYTES:
        raise ValueError(
            f"ZIP {source!r} declares {total / 1e9:.1f} GB decompressed "
            f"(cap {_MAX_ZIP_EXTRACT_BYTES / 1e9:.0f} GB) — refusing to extract."
        )


def _safe_extract_zip(zip_path: Path, out_dir: Path) -> int:
    """Extract a ZIP after validating total size and member paths. Returns member count."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        check_zip_size(zf, zip_path.name)
        resolved_out_dir = out_dir.resolve()
        for member in zf.infolist():
            target = (resolved_out_dir / member.filename).resolve()
            # Path comparison, not string prefixing: a hardcoded "/" separator
            # rejects every legitimate member on Windows, where the resolved
            # target is separated by "\" and nothing ever matches the prefix.
            if target == resolved_out_dir or not target.is_relative_to(resolved_out_dir):
                raise ValueError(f"Unsafe path in ZIP: {member.filename!r}")
        zf.extractall(out_dir)
        return len(zf.namelist())


# --- C6 climate normals ZIP ---


def download_climate_normals_zip(output_dir: Path, force: bool = False, *, fetcher: Fetcher) -> DownloadResult:
    """Download C6 climate normals ZIP from opendata.swiss and extract."""
    out_dir = output_dir / "climate_normals"
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / "normwerte.zip"

    # Skip on the *extraction output*, not the ZIP: a run that died between
    # download and extract must not be mistaken for a completed one.
    if not force and any(out_dir.glob("*.txt")):
        logger.info("  Climate normals already downloaded and extracted — skipping")
        return DownloadResult(total_assets=1, downloaded=0, skipped=1, filenames=[])

    logger.info("%s", "=" * 60)
    logger.info("Climate normals (C6): downloading from opendata.swiss")
    logger.info("%s", "=" * 60)

    fetcher.stream(CLIMATE_NORMALS_ZIP_URL, filepath, timeout=120)
    logger.info("  Downloaded: %s (%.0f KB)", filepath.name, filepath.stat().st_size / 1024)

    extracted = _safe_extract_zip(filepath, out_dir)
    logger.info("  Extracted %d files", extracted)

    return DownloadResult(total_assets=1, downloaded=1, skipped=0, filenames=["normwerte.zip"])


# --- Indoor climate scenarios ZIP (single .csv.zip of per-station CSVs) ---


def download_climate_scenarios_indoor(
    output_dir: Path,
    collection_key: str = "climate_scenarios_indoor",
    force: bool = False,
    *,
    fetcher: Fetcher,
) -> DownloadResult:
    """Download and extract the indoor climate scenarios ZIP.

    The collection ships a single ``.csv.zip`` (per-station, per-scenario hourly
    CSVs) rather than individual STAC CSV assets, so it needs its own download
    path. The archive is fetched from the STAC API and its members extracted to
    ``output_dir/<collection_key>/``.
    """
    collection_id = COLLECTIONS[collection_key]
    out_dir = output_dir / collection_key
    out_dir.mkdir(parents=True, exist_ok=True)

    if not force and any(out_dir.glob("*.csv")):
        logger.info("  %s already extracted — skipping", collection_key)
        return DownloadResult(total_assets=1, downloaded=0, skipped=1, filenames=[])

    items = fetcher.items(collection_id)
    archives = assets_of(items, suffixes=(".zip",))
    archive = archives[0] if archives else None
    if archive is None:
        logger.warning("  No .zip asset found for %s", collection_key)
        return DownloadResult()

    logger.info("%s", "=" * 60)
    logger.info("Indoor scenarios (%s): downloading ZIP", collection_id)
    logger.info("%s", "=" * 60)

    zip_path = out_dir / archive.name
    fetcher.stream(archive.href, zip_path, timeout=300)
    logger.info("  Downloaded: %s (%.1f MB)", zip_path.name, zip_path.stat().st_size / 1e6)

    extracted = _safe_extract_zip(zip_path, out_dir)
    logger.info("  Extracted %d files", extracted)

    return DownloadResult(total_assets=1, downloaded=1, skipped=0, filenames=[zip_path.name])
