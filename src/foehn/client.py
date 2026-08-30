"""Download MeteoSwiss data from the STAC API and opendata.swiss."""

from __future__ import annotations

import json
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from foehn._urls import asset_filename, clean_href
from foehn.collections import (
    CLIMATE_NORMALS_ZIP_URL,
    COLLECTIONS,
    DatasetKind,
    forecast_run_from_filename,
    kind,
    time_slice_from_filename,
)
from foehn.convert import utf8_meteoswiss_csv
from foehn.fetch import DEFAULT_WORKERS, Fetcher

logger = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    """Summary of a download call. Returned by all download_* functions.

    Callers use this to decide whether to run expensive downstream work
    (e.g. Spark MERGE INTO) without scanning the output directory.

    ``total_assets`` is per-producer: ``download_collection`` counts the assets
    left after its time-slice filter, while ``download_grib2``/``download_netcdf``
    count every matching asset in the listing. ``foehn.download`` sums results
    across the metadata and data passes, so treat the total as a scale hint
    rather than a figure comparable between dataset types.
    """

    total_assets: int = 0
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    filenames: list[str] = field(default_factory=list)


def _dedupe_by_destination(pairs: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    """Keep one (href, filepath) pair per destination path.

    Two hrefs resolving to the same local filename would have workers streaming
    into the same ``.part`` file at once and corrupting it. A collision is an
    upstream naming quirk we can't resolve here, so keep the first and say so
    rather than dropping one silently.
    """
    seen: dict[Path, str] = {}
    for href, filepath in pairs:
        if filepath in seen:
            logger.warning("  Duplicate destination %s — skipping %s", filepath.name, href)
            continue
        seen[filepath] = href
    return [(href, filepath) for filepath, href in seen.items()]


# --- State files (ETags + last-run timestamp) ---


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text via a sibling temp file + Path.replace so readers never see a torn write."""
    _atomic_write_bytes(path, text.encode("utf-8"))


def _atomic_write_bytes(path: Path, data: bytes | memoryview) -> None:
    """Write bytes via a sibling temp file + Path.replace so readers never see a torn write."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


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
    _atomic_write_text(path, json.dumps(etags, indent=2))


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
    _atomic_write_text(path, json.dumps({"timestamp": datetime.now(UTC).isoformat()}))


# --- CSV downloads ---


def _download_csv(
    fetcher: Fetcher,
    href: str,
    filepath: Path,
    old_etag: str | None,
) -> tuple[str, str, str, str | None]:
    """Fetch a single CSV, re-encode to UTF-8, and write it to disk.

    Returns (status, href, filename, new_etag) where status is "downloaded" or "skipped".
    Only sends the stored ETag when the local file is still there — otherwise a
    304 would report "skipped" over a file that no longer exists.
    """
    filename = filepath.name
    etag = old_etag if old_etag and filepath.exists() else None
    fetched = fetcher.get(href, etag=etag, timeout=60)
    if fetched.not_modified:
        return ("skipped", href, filename, None)
    # MeteoSwiss CSVs are usually UTF-8 but some are Windows-1252; normalise to UTF-8.
    # Bytes in, bytes out — decoding to str here only to re-encode on write would
    # hold three copies of a multi-hundred-MB CSV at once, times ``workers``.
    _atomic_write_bytes(filepath, utf8_meteoswiss_csv(fetched.body))
    return ("downloaded", href, filename, fetched.etag)


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
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # Collect matching CSV assets. ``all_csv_hrefs`` is the full pre-filter set
    # (every CSV in the listing, regardless of time slice) — the prune universe
    # below, so ETags for slices outside this run's data_types are kept.
    skip_data_type_filter = kind(collection_key) is DatasetKind.FORECAST_CSV
    csv_assets = []
    all_csv_hrefs: set[str] = set()
    for item in items:
        assets = item.get("assets", {})
        for asset_info in assets.values():
            href = asset_info.get("href", "")
            clean = clean_href(href)
            if not clean.endswith(".csv"):
                continue
            all_csv_hrefs.add(href)
            if not skip_data_type_filter:
                slice_ = time_slice_from_filename(clean)
                if slice_ is not None and slice_ not in data_types:
                    continue
            csv_assets.append((href, asset_info))

    # One forecast run is ~32 files at ~30 MB each (~1 GB); the full retained
    # window is ~40 runs (~40 GB). Keep only the newest complete-ish run.
    if kind(collection_key) is DatasetKind.FORECAST_CSV and csv_assets:
        runs = {run for href, _ in csv_assets if (run := forecast_run_from_filename(clean_href(href))) is not None}
        if runs:
            latest_run = max(runs)
            csv_assets = [
                (href, info) for href, info in csv_assets if forecast_run_from_filename(clean_href(href)) == latest_run
            ]
            logger.info("  Latest forecast run: %s (of %d available)", latest_run, len(runs))

    logger.info("  %d CSV files to process", len(csv_assets))

    def _do_csv(href: str, filepath: Path, etag: str | None) -> tuple[str, str, str, str | None]:
        return _download_csv(fetcher, href, filepath, etag)

    total = len(csv_assets)
    downloaded = 0
    skipped = 0
    failed = 0
    filenames: list[str] = []
    fetch_targets = _dedupe_by_destination([(href, out_dir / asset_filename(href)) for href, _ in csv_assets])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_href = {
            pool.submit(_do_csv, href, filepath, etags.get(href)): href for href, filepath in fetch_targets
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
            else:
                # Server returned 200 without an ETag. Keeping the old one would
                # re-send it as If-None-Match forever, so this asset would
                # re-download every run and never once be skipped.
                etags.pop(href, None)
            downloaded += 1
            filenames.append(filename)
            logger.info("  [%d/%d] Downloaded: %s", i, total, filename)

    # Prune ETags for assets that no longer exist upstream, so _etags.json
    # doesn't grow forever (e.g. forecast runs get fresh filenames each cycle).
    # Only on a clean full listing: with ``since`` the item list is partial, and
    # after failures the universe may be incomplete.
    if since is None and failed == 0:
        prefix = f"/{collection_id}/"
        stale = [k for k in etags if prefix in k and k not in all_csv_hrefs]
        for k in stale:
            del etags[k]
        if stale:
            logger.info("  Pruned %d stale ETag entries", len(stale))

    save_etags(state_dir, etags)
    if skipped:
        logger.info("  Skipped %d unchanged files", skipped)
    if failed:
        logger.warning("  %d file(s) failed to download", failed)
    logger.info("  Done — %d files downloaded", downloaded)

    return DownloadResult(
        total_assets=total, downloaded=downloaded, skipped=skipped, failed=failed, filenames=filenames
    )


# --- Metadata downloads ---


def download_metadata(
    collection_key: str, output_dir: Path, workers: int = DEFAULT_WORKERS, *, fetcher: Fetcher
) -> DownloadResult:
    """Download collection-level metadata files (stations, parameters, inventory)."""
    collection_id = COLLECTIONS[collection_key]
    out_dir = output_dir / collection_key
    out_dir.mkdir(parents=True, exist_ok=True)

    coll = fetcher.collection(collection_id)
    assets = coll.get("assets", {})
    if not assets:
        return DownloadResult()

    targets = _dedupe_by_destination(
        [
            (asset_info["href"], out_dir / asset_filename(asset_info["href"]))
            for asset_info in assets.values()
            if clean_href(asset_info.get("href", "")).endswith(".csv")
        ]
    )
    if not targets:
        return DownloadResult()

    def _do_csv(href: str, filepath: Path) -> tuple[str, str, str, str | None]:
        return _download_csv(fetcher, href, filepath, None)

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
        remote_dt = datetime.fromisoformat(remote_updated)
        local_dt = datetime.fromtimestamp(filepath.stat().st_mtime, tz=UTC)
    except (ValueError, OSError):
        return False
    return remote_dt > local_dt


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
    out_dir = output_dir / collection_key
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("%s", "=" * 60)
    logger.info("GRIB2 Collection: %s", collection_id)
    logger.info("Output dir: %s", out_dir)
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

    binary_assets = []
    for item in items:
        item_updated = item.get("properties", {}).get("updated", "")
        assets = item.get("assets", {})
        for asset_info in assets.values():
            href = asset_info.get("href", "")
            clean = clean_href(href)
            # Accept grib2, h5, and other binary formats
            if any(clean.endswith(ext) for ext in (".grib2", ".h5", ".hdf5")):
                # Per-asset "updated" is preferred when present (STAC allows it),
                # otherwise fall back to the item-level updated timestamp.
                updated = asset_info.get("updated") or item_updated
                binary_assets.append((href, clean.split("/")[-1], updated))

    logger.info("  %d binary files to download", len(binary_assets))

    to_fetch = _dedupe_by_destination(
        [
            (href, out_dir / filename)
            for href, filename, updated in binary_assets
            if _needs_redownload(out_dir / filename, updated)
        ]
    )

    def _do_binary(href: str, filepath: Path) -> str:
        fetcher.stream(href, filepath)
        return filepath.name

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
    out_dir = output_dir / collection_key
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("%s", "=" * 60)
    logger.info("NetCDF Collection: %s", collection_id)
    logger.info("Output dir: %s", out_dir)
    logger.info("%s", "=" * 60)

    items = fetcher.items(collection_id)
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
            clean = clean_href(href)
            if not clean.endswith((".nc", ".tif", ".zip")):
                continue
            total_assets += 1
            filepath = out_dir / clean.split("/")[-1]
            if filepath.exists():
                continue
            targets.append((href, filepath))
    targets = _dedupe_by_destination(targets)

    def _do_binary(href: str, filepath: Path) -> str:
        fetcher.stream(href, filepath)
        return filepath.name

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


# --- ZIP safety (zip-slip + decompression-bomb guards) ---

# Cap on declared total decompressed size. Generous — the largest legitimate
# archive (indoor scenarios) is well under this — but stops a decompression
# bomb from a compromised upstream filling the disk (or RAM, for in-memory
# reads). Python's zipfile enforces each member's declared size on read, so
# checking the headers is sufficient.
_MAX_ZIP_EXTRACT_BYTES = 10 * 1024**3  # 10 GiB


def _check_zip_size(zf: zipfile.ZipFile, source: str) -> None:
    """Raise ValueError if the archive declares more decompressed bytes than the cap."""
    total = sum(m.file_size for m in zf.infolist())
    if total > _MAX_ZIP_EXTRACT_BYTES:
        raise ValueError(
            f"ZIP {source!r} declares {total / 1e9:.1f} GB decompressed "
            f"(cap {_MAX_ZIP_EXTRACT_BYTES / 1e9:.0f} GB) — refusing to extract."
        )


def _safe_extract_zip(zip_path: Path, out_dir: Path) -> int:
    """Extract a ZIP after validating total size and member paths. Returns member count."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        _check_zip_size(zf, zip_path.name)
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
        logger.warning("  No .zip asset found for %s", collection_key)
        return DownloadResult()

    logger.info("%s", "=" * 60)
    logger.info("Indoor scenarios (%s): downloading ZIP", collection_id)
    logger.info("%s", "=" * 60)

    zip_path = out_dir / asset_filename(zip_href)
    fetcher.stream(zip_href, zip_path, timeout=300)
    logger.info("  Downloaded: %s (%.1f MB)", zip_path.name, zip_path.stat().st_size / 1e6)

    extracted = _safe_extract_zip(zip_path, out_dir)
    logger.info("  Extracted %d files", extracted)

    return DownloadResult(total_assets=1, downloaded=1, skipped=0, filenames=[zip_path.name])
