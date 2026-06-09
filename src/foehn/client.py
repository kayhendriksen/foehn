"""Download MeteoSwiss data from the STAC API and opendata.swiss."""

from __future__ import annotations

import json
import logging
import os
import threading
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from foehn._urls import validate_download_href
from foehn.collections import (
    CLIMATE_NORMALS_ZIP_URL,
    COLLECTIONS,
    FORECAST_CSV_COLLECTIONS,
    STAC_API_BASE,
    time_slice_from_filename,
)
from foehn.convert import decode_meteoswiss_csv
from foehn.stac import get_collection_items, get_collection_metadata

logger = logging.getLogger(__name__)

# Default concurrency for per-asset downloads. Kept modest to stay polite on the
# MeteoSwiss/CSCS CDNs — they handle bursts fine but we don't need to hammer them.
DEFAULT_WORKERS = 8


@dataclass
class DownloadResult:
    """Summary of a download call. Returned by all download_* functions.

    Callers use this to decide whether to run expensive downstream work
    (e.g. Spark MERGE INTO) without scanning the output directory.
    """

    total_assets: int = 0
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    filenames: list[str] = field(default_factory=list)


def _retry_session(
    retries: int = 3,
    backoff_factor: float = 1.0,
    status_forcelist: tuple[int, ...] = (500, 502, 503, 504),
    pool_maxsize: int = DEFAULT_WORKERS,
) -> requests.Session:
    """Return a requests session with automatic retry on transient errors."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["GET"],
        raise_on_status=False,
        connect=retries,
        read=retries,
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=pool_maxsize,
        pool_maxsize=pool_maxsize,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _thread_local_session() -> Callable[[], requests.Session]:
    """Return a getter that lazily creates one ``requests.Session`` per worker thread.

    ``requests.Session`` is not fully thread-safe (cookie jar, headers), so any
    parallel download path should use this rather than sharing a single session
    across a ``ThreadPoolExecutor``. Sessions live for the lifetime of the
    returned closure — when the caller drops the getter, the per-thread sessions
    are garbage-collected with it.
    """
    local = threading.local()

    def get() -> requests.Session:
        if not hasattr(local, "session"):
            local.session = _retry_session(pool_maxsize=1)
        return local.session

    return get


# --- State files (ETags + last-run timestamp) ---


def load_etags(data_dir: Path) -> dict:
    path = data_dir / "_etags.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_etags(data_dir: Path, etags: dict):
    path = data_dir / "_etags.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(etags, indent=2))


def load_last_run(data_dir: Path) -> str | None:
    """Return ISO timestamp of last successful run, or None."""
    path = data_dir / "_last_run.json"
    if path.exists():
        data = json.loads(path.read_text())
        return data.get("timestamp")
    return None


def save_last_run(data_dir: Path):
    path = data_dir / "_last_run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"timestamp": datetime.now(UTC).isoformat()}))


# --- CSV downloads ---


def _download_csv(
    session: requests.Session,
    href: str,
    filepath: Path,
    old_etag: str | None,
) -> tuple[str, str, str, str | None]:
    """Fetch a single CSV, re-encode to UTF-8, and write it to disk.

    Returns (status, href, filename, new_etag) where status is "downloaded" or "skipped".
    """
    validate_download_href(href)
    filename = filepath.name
    headers = {"If-None-Match": old_etag} if old_etag and filepath.exists() else {}
    resp = session.get(href, headers=headers, timeout=60)
    if resp.status_code == 304:
        return ("skipped", href, filename, None)
    resp.raise_for_status()
    # MeteoSwiss CSVs are usually UTF-8 but some are Windows-1252; normalise to UTF-8.
    content = decode_meteoswiss_csv(resp.content)
    filepath.write_text(content, encoding="utf-8")
    return ("downloaded", href, filename, resp.headers.get("ETag"))


def download_collection(
    collection_key: str,
    output_dir: Path,
    data_types: list[str] | None = None,
    since: str | None = None,
    workers: int = DEFAULT_WORKERS,
) -> DownloadResult:
    """Download CSVs for a collection.

    Args:
        collection_key: Key from COLLECTIONS (e.g. "smn").
        output_dir: Root directory for bronze downloads (files go to output_dir/<key>/).
        data_types: List of "historical", "recent", "now". Defaults to ["recent"].
        since: ISO timestamp — only process items updated after this time.
        workers: Number of concurrent HTTP downloads.

    Returns:
        DownloadResult with counts and list of newly downloaded filenames.
    """
    if data_types is None:
        data_types = ["recent"]

    collection_id = COLLECTIONS[collection_key]
    out_dir = output_dir / collection_key
    out_dir.mkdir(parents=True, exist_ok=True)

    etags = load_etags(output_dir.parent)

    logger.info("%s", "=" * 60)
    logger.info("Collection: %s", collection_id)
    logger.info("Data types: %s", data_types)
    logger.info("Output dir: %s", out_dir)
    logger.info("%s", "=" * 60)

    items = get_collection_items(collection_id)
    logger.info("  Found %d items", len(items))

    # Filter to items updated since last run
    if since:
        items = [item for item in items if item.get("properties", {}).get("updated", "") > since]
        logger.info("  %d items updated since last run", len(items))
        if not items:
            logger.info("  Nothing changed — skipping")
            return DownloadResult()

    # For forecast collections, only keep the latest item (newest forecast run)
    if collection_key in FORECAST_CSV_COLLECTIONS and items:
        items.sort(
            key=lambda x: x.get("properties", {}).get("datetime", x.get("id", "")),
            reverse=True,
        )
        items = items[:1]
        logger.info("  Using latest forecast: %s", items[0].get("id", "?"))

    # Collect matching CSV assets
    skip_data_type_filter = collection_key in FORECAST_CSV_COLLECTIONS
    csv_assets = []
    for item in items:
        assets = item.get("assets", {})
        for asset_info in assets.values():
            href = asset_info.get("href", "")
            clean = href.split("?", 1)[0]
            if not clean.endswith(".csv"):
                continue
            if not skip_data_type_filter:
                slice_ = time_slice_from_filename(clean)
                if slice_ is not None and slice_ not in data_types:
                    continue
            csv_assets.append((href, asset_info))

    logger.info("  %d CSV files to process", len(csv_assets))

    get_session = _thread_local_session()

    def _do_csv(href: str, filepath: Path, etag: str | None) -> tuple[str, str, str, str | None]:
        return _download_csv(get_session(), href, filepath, etag)

    total = len(csv_assets)
    downloaded = 0
    skipped = 0
    failed = 0
    filenames: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_href = {
            pool.submit(
                _do_csv,
                href,
                out_dir / href.split("?")[0].split("/")[-1],
                etags.get(href),
            ): href
            for href, _ in csv_assets
        }
        for i, fut in enumerate(as_completed(future_to_href), 1):
            # Isolate per-asset failures: one transient 404/timeout must not abort
            # the batch (and discard the ETags collected from the other assets).
            try:
                status, href, filename, new_etag = fut.result()
            except Exception as exc:
                failed += 1
                logger.warning("  [%d/%d] FAILED: %s — %s", i, total, future_to_href[fut], exc)
                continue
            if status == "skipped":
                skipped += 1
                continue
            if new_etag:
                etags[href] = new_etag
            downloaded += 1
            filenames.append(filename)
            logger.info("  [%d/%d] Downloaded: %s", i, total, filename)

    save_etags(output_dir.parent, etags)
    if skipped:
        logger.info("  Skipped %d unchanged files", skipped)
    if failed:
        logger.warning("  %d file(s) failed to download", failed)
    logger.info("  Done — %d files downloaded", downloaded)

    return DownloadResult(
        total_assets=total, downloaded=downloaded, skipped=skipped, failed=failed, filenames=filenames
    )


# --- Metadata downloads ---


def download_metadata(collection_key: str, output_dir: Path, workers: int = DEFAULT_WORKERS) -> DownloadResult:
    """Download collection-level metadata files (stations, parameters, inventory)."""
    collection_id = COLLECTIONS[collection_key]
    out_dir = output_dir / collection_key
    out_dir.mkdir(parents=True, exist_ok=True)

    coll = get_collection_metadata(collection_id)
    assets = coll.get("assets", {})
    if not assets:
        return DownloadResult()

    targets = [
        (asset_info["href"], out_dir / asset_info["href"].split("?")[0].split("/")[-1])
        for asset_info in assets.values()
        if asset_info.get("href", "").split("?", 1)[0].endswith(".csv")
    ]
    if not targets:
        return DownloadResult()

    get_session = _thread_local_session()

    def _do_csv(href: str, filepath: Path) -> tuple[str, str, str, str | None]:
        return _download_csv(get_session(), href, filepath, None)

    downloaded = 0
    failed = 0
    filenames: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_href = {pool.submit(_do_csv, href, filepath): href for href, filepath in targets}
        for fut in as_completed(future_to_href):
            try:
                filename = fut.result()[2]
            except Exception as exc:
                failed += 1
                logger.warning("  Metadata FAILED: %s — %s", future_to_href[fut], exc)
                continue
            downloaded += 1
            filenames.append(filename)
            logger.info("  Metadata: %s", filename)

    if downloaded:
        logger.info("  %d metadata files downloaded", downloaded)
    if failed:
        logger.warning("  %d metadata file(s) failed to download", failed)

    return DownloadResult(total_assets=len(targets), downloaded=downloaded, failed=failed, filenames=filenames)


# --- GRIB2 / HDF5 downloads ---


def _needs_redownload(filepath: Path, remote_updated: str) -> bool:
    """Decide whether to (re)download a binary asset.

    Handles MeteoSwiss's in-place overwrites — e.g. CombiPrecip reanalysis
    (CPCH) replaces the original CPC hourly file with the same filename
    ~8 days later. A plain exists() check would leave the stale version
    on disk; comparing the STAC "updated" timestamp against local mtime
    picks up those server-side updates.
    """
    if not filepath.exists():
        return True
    if not remote_updated:
        return False
    try:
        remote_dt = datetime.fromisoformat(remote_updated.replace("Z", "+00:00"))
        local_dt = datetime.fromtimestamp(filepath.stat().st_mtime, tz=UTC)
    except (ValueError, OSError):
        return False
    return remote_dt > local_dt


def _download_binary(session: requests.Session, href: str, filepath: Path, timeout: int = 120) -> str:
    """Stream a binary asset to disk atomically. Returns the filename.

    Streams into a sibling ``.part`` file and only ``os.replace``s it onto the
    final path once the body is fully written. A timeout or connection drop
    mid-stream therefore leaves no file at ``filepath`` — so the next run's
    existence/mtime check won't mistake a truncated download for a complete one.
    """
    validate_download_href(href)
    tmp = filepath.with_name(filepath.name + ".part")
    try:
        with session.get(href, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
        os.replace(tmp, filepath)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return filepath.name


def download_grib2(
    collection_key: str,
    output_dir: Path,
    since: str | None = None,
    workers: int = DEFAULT_WORKERS,
) -> DownloadResult:
    """Download GRIB2/HDF5 binary files (latest page only)."""
    collection_id = COLLECTIONS[collection_key]
    out_dir = output_dir / collection_key
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("%s", "=" * 60)
    logger.info("GRIB2 Collection: %s", collection_id)
    logger.info("Output dir: %s", out_dir)
    logger.info("%s", "=" * 60)

    # Only fetch first page — forecast/radar data is ephemeral
    url = f"{STAC_API_BASE}/collections/{collection_id}/items?limit=100"
    with _retry_session() as session:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        items = resp.json().get("features", [])
    logger.info("  Found %d items (latest page)", len(items))

    if since:
        items = [item for item in items if item.get("properties", {}).get("updated", "") > since]
        logger.info("  %d items updated since last run", len(items))
        if not items:
            logger.info("  Nothing changed — skipping")
            return DownloadResult()

    binary_assets = []
    for item in items:
        item_updated = item.get("properties", {}).get("updated", "")
        assets = item.get("assets", {})
        for asset_info in assets.values():
            href = asset_info.get("href", "")
            clean = href.split("?")[0]
            # Accept grib2, h5, and other binary formats
            if any(clean.endswith(ext) for ext in (".grib2", ".h5", ".hdf5")):
                # Per-asset "updated" is preferred when present (STAC allows it),
                # otherwise fall back to the item-level updated timestamp.
                updated = asset_info.get("updated") or item_updated
                binary_assets.append((href, clean.split("/")[-1], updated))

    logger.info("  %d binary files to download", len(binary_assets))

    to_fetch = [
        (href, out_dir / filename)
        for href, filename, updated in binary_assets
        if _needs_redownload(out_dir / filename, updated)
    ]

    get_session = _thread_local_session()

    def _do_binary(href: str, filepath: Path) -> str:
        return _download_binary(get_session(), href, filepath)

    total = len(to_fetch)
    downloaded = 0
    failed = 0
    filenames: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_href = {pool.submit(_do_binary, href, filepath): href for href, filepath in to_fetch}
        for i, fut in enumerate(as_completed(future_to_href), 1):
            try:
                filename = fut.result()
            except Exception as exc:
                failed += 1
                logger.warning("  [%d/%d] FAILED: %s — %s", i, total, future_to_href[fut], exc)
                continue
            downloaded += 1
            filenames.append(filename)
            logger.info("  [%d/%d] Downloaded: %s", i, total, filename)

    if failed:
        logger.warning("  %d binary file(s) failed to download", failed)
    logger.info("  Done — %d binary files downloaded", downloaded)

    return DownloadResult(
        total_assets=len(binary_assets),
        downloaded=downloaded,
        skipped=len(binary_assets) - total,
        failed=failed,
        filenames=filenames,
    )


# --- NetCDF / GeoTIFF / ZIP downloads ---


def download_netcdf(
    collection_key: str,
    output_dir: Path,
    since: str | None = None,
    workers: int = DEFAULT_WORKERS,
) -> DownloadResult:
    """Download NetCDF, GeoTIFF, and ZIP files for spatial/static collections.

    Args:
        collection_key: Key from COLLECTIONS.
        output_dir: Root directory for bronze downloads.
        since: ISO timestamp — only process items updated after this time.
        workers: Number of concurrent HTTP downloads.
    """
    collection_id = COLLECTIONS[collection_key]
    out_dir = output_dir / collection_key
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("%s", "=" * 60)
    logger.info("NetCDF Collection: %s", collection_id)
    logger.info("Output dir: %s", out_dir)
    logger.info("%s", "=" * 60)

    items = get_collection_items(collection_id, require_csv=False)
    logger.info("  Found %d items", len(items))

    if since:
        items = [item for item in items if item.get("properties", {}).get("updated", "") > since]
        logger.info("  %d items updated since last run", len(items))
        if not items:
            logger.info("  Nothing changed — skipping")
            return DownloadResult()

    total_assets = 0
    targets: list[tuple[str, Path]] = []
    for item in items:
        for asset_info in item.get("assets", {}).values():
            href = asset_info.get("href", "")
            clean = href.split("?")[0]
            if not clean.endswith((".nc", ".tif", ".zip")):
                continue
            total_assets += 1
            filepath = out_dir / clean.split("/")[-1]
            if filepath.exists():
                continue
            targets.append((href, filepath))

    get_session = _thread_local_session()

    def _do_binary(href: str, filepath: Path) -> str:
        return _download_binary(get_session(), href, filepath)

    downloaded = 0
    failed = 0
    filenames: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_href = {pool.submit(_do_binary, href, filepath): href for href, filepath in targets}
        for fut in as_completed(future_to_href):
            try:
                filename = fut.result()
            except Exception as exc:
                failed += 1
                logger.warning("  FAILED: %s — %s", future_to_href[fut], exc)
                continue
            downloaded += 1
            filenames.append(filename)
            logger.info("  Downloaded: %s", filename)

    if failed:
        logger.warning("  %d file(s) failed to download", failed)
    logger.info("  Done — %d files downloaded", downloaded)

    # ``skipped`` is the count of pre-existing files left untouched (total assets
    # minus the ones we queued); failures are tracked separately in ``failed``.
    return DownloadResult(
        total_assets=total_assets,
        downloaded=downloaded,
        skipped=total_assets - len(targets),
        failed=failed,
        filenames=filenames,
    )


# --- C6 climate normals ZIP ---


def download_climate_normals_zip(output_dir: Path, force: bool = False) -> DownloadResult:
    """Download C6 climate normals ZIP from opendata.swiss and extract."""
    out_dir = output_dir / "climate_normals"
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / "normwerte.zip"

    if filepath.exists() and not force:
        logger.info("  Climate normals ZIP already downloaded — skipping")
        return DownloadResult(total_assets=1, downloaded=0, skipped=1, filenames=[])

    logger.info("%s", "=" * 60)
    logger.info("Climate normals (C6): downloading from opendata.swiss")
    logger.info("%s", "=" * 60)

    with _retry_session() as session:
        resp = session.get(CLIMATE_NORMALS_ZIP_URL, timeout=120)
        resp.raise_for_status()
    filepath.write_bytes(resp.content)
    logger.info("  Downloaded: normwerte.zip (%.0f KB)", len(resp.content) / 1024)

    with zipfile.ZipFile(filepath, "r") as zf:
        resolved_out_dir = out_dir.resolve()
        for member in zf.infolist():
            target = (resolved_out_dir / member.filename).resolve()
            if not str(target).startswith(str(resolved_out_dir) + "/"):
                raise ValueError(f"Unsafe path in ZIP: {member.filename!r}")
        zf.extractall(out_dir)
        logger.info("  Extracted %d files", len(zf.namelist()))

    return DownloadResult(total_assets=1, downloaded=1, skipped=0, filenames=["normwerte.zip"])


# --- Indoor climate scenarios ZIP (single .csv.zip of per-station CSVs) ---


def download_climate_scenarios_indoor(
    output_dir: Path,
    collection_key: str = "climate_scenarios_indoor",
    force: bool = False,
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

    items = get_collection_items(collection_id, require_csv=False, verbose=False)
    zip_href = next(
        (
            asset_info.get("href", "")
            for item in items
            for asset_info in item.get("assets", {}).values()
            if asset_info.get("href", "").split("?")[0].endswith(".zip")
        ),
        None,
    )
    if not zip_href:
        logger.warning("  No .zip asset found for %s", collection_key)
        return DownloadResult()

    logger.info("%s", "=" * 60)
    logger.info("Indoor scenarios (%s): downloading ZIP", collection_id)
    logger.info("%s", "=" * 60)

    validate_download_href(zip_href)
    zip_path = out_dir / zip_href.split("?")[0].split("/")[-1]
    with _retry_session() as session:
        resp = session.get(zip_href, timeout=300)
        resp.raise_for_status()
    zip_path.write_bytes(resp.content)
    logger.info("  Downloaded: %s (%.1f MB)", zip_path.name, len(resp.content) / 1e6)

    with zipfile.ZipFile(zip_path, "r") as zf:
        resolved_out_dir = out_dir.resolve()
        for member in zf.infolist():
            target = (resolved_out_dir / member.filename).resolve()
            if not str(target).startswith(str(resolved_out_dir) + "/"):
                raise ValueError(f"Unsafe path in ZIP: {member.filename!r}")
        zf.extractall(out_dir)
        logger.info("  Extracted %d files", len(zf.namelist()))

    return DownloadResult(total_assets=1, downloaded=1, skipped=0, filenames=[zip_path.name])
