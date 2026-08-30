"""Download MeteoSwiss data from the STAC API and opendata.swiss."""

from __future__ import annotations

import json
import logging
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from foehn.assets import Asset, assets_of, collection_assets, latest_run_of, select
from foehn.collections import CLIMATE_NORMALS_ZIP_URL, COLLECTIONS
from foehn.fetch import DEFAULT_WORKERS, Fetcher
from foehn.transfer import (
    DownloadResult,
    SkipRule,
    Writer,
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


# --- One listing path, configured per dataset kind ---


def _banner(title: str, out_dir: Path) -> None:
    """The four download paths each printed this by hand, ten lines in total."""
    logger.info("%s", "=" * 60)
    logger.info("%s", title)
    logger.info("Output dir: %s", out_dir)
    logger.info("%s", "=" * 60)


def _updated_since(items: list[dict], since: str | None) -> list[dict]:
    """Keep only the items MeteoSwiss restated after *since*.

    Written out verbatim in three of the four download paths, along with its two
    log lines. What ``since`` means is one rule.
    """
    if not since:
        return items
    kept = [item for item in items if item.get("properties", {}).get("updated", "") > since]
    logger.info("  %d items updated since last run", len(kept))
    return kept


def _prune_stale_etags(etags: dict, collection_id: str, listed: set[str]) -> None:
    """Drop ETags for assets that no longer exist upstream.

    Otherwise ``_etags.json`` grows forever — forecast runs get fresh filenames
    every cycle. Only safe on a clean, complete listing: see the call site.
    """
    prefix = f"/{collection_id}/"
    stale = [k for k in etags if prefix in k and k not in listed]
    for k in stale:
        del etags[k]
    if stale:
        logger.info("  Pruned %d stale ETag entries", len(stale))


def stac_download(
    *,
    suffixes: tuple[str, ...],
    title: str,
    label: str = "file",
    write: Writer = stream_to_disk,
    skip: SkipRule | None = None,
    etags: bool = False,
    max_items: int | None = None,
    time_sliced: bool = False,
    latest_run: bool = False,
    with_metadata: bool = False,
):
    """Build the download adapter for a kind whose assets come from a STAC listing.

    ``download_collection``, ``download_grib2`` and ``download_netcdf`` were the
    same body three times — list, drop what has not been restated since last run,
    pick the assets, hand them to :func:`~foehn.transfer.fetch_all`. Only the
    arguments below ever differed, and two of them were being re-derived from the
    dataset key inside the loop rather than stated by the kind: whether to keep
    only the newest forecast run, and whether the metadata pass runs first.

    Returns a :class:`~foehn.registry.DownloadAdapter`, so the odd kinds that do
    not list STAC at all (the ZIP-shipped ones) stay expressible as plain
    functions of the same shape.

    Args:
        suffixes: Which asset file types this kind downloads.
        title: Heading for the log banner, e.g. "GRIB2 collection".
        label: Noun for the per-file log lines ("CSV", "binary file").
        write: How one asset becomes one file — conditional GET for CSV,
            straight to disk for binaries.
        skip: Decided locally before anything is queued.
        etags: Whether this kind keeps an ETag store (the conditional-GET path).
        max_items: Stop paginating early. The ephemeral collections only ever
            want the newest page.
        time_sliced: Whether ``time_slice`` narrows this kind's assets. False
            where the filenames carry no slice.
        latest_run: Keep only the newest **Forecast run**. One run is ~32 files
            at ~30 MB, and the retained window is ~40 of them.
        with_metadata: Whether the collection-level metadata files are fetched
            in the same pass and reported as one result.
    """

    def download(
        dataset: str,
        bronze_dir: Path,
        *,
        time_slice: list[str],
        since: str | None = None,
        workers: int = DEFAULT_WORKERS,
        fetcher: Fetcher,
        **_: object,
    ) -> DownloadResult:
        result = DownloadResult()
        if with_metadata:
            result = download_metadata(dataset, bronze_dir, workers=workers, fetcher=fetcher)

        collection_id = COLLECTIONS[dataset]
        out_dir = bronze_dir / dataset
        _banner(f"{title}: {collection_id}", out_dir)

        items = fetcher.items(collection_id, max_items=max_items)
        logger.info("  Found %d items%s", len(items), " (latest page)" if max_items else "")

        items = _updated_since(items, since)
        if since and not items:
            logger.info("  Nothing changed — skipping")
            return result

        # ``every`` is the full pre-filter set — the prune universe below, so
        # ETags for slices outside this run's time_slice are kept.
        every = assets_of(items, suffixes=suffixes)
        wanted = select(
            every,
            time_slices=time_slice if time_sliced else None,
            latest_run=latest_run,
        )
        if latest_run and (run := latest_run_of(every)) is not None:
            available = len({a.forecast_run for a in every if a.forecast_run is not None})
            logger.info("  Latest forecast run: %s (of %d available)", run, available)

        # The conditional-GET writer decides "unchanged" from the server's 304,
        # so the ETag store is this kind's skip rule.
        store = load_etags(bronze_dir.parent) if etags else None
        fetched = fetch_all(
            wanted,
            out_dir,
            fetcher=fetcher,
            workers=workers,
            write=write,
            skip=skip,
            etags=store,
            label=label,
        )

        if store is not None:
            # Only on a clean full listing: with ``since`` the item list is
            # partial, and after failures the universe may be incomplete.
            if since is None and fetched.failed == 0:
                _prune_stale_etags(store, collection_id, {a.href for a in every})
            save_etags(bronze_dir.parent, store)

        return result + fetched

    return download


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


# --- Skip rules ---


def already_current(asset: Asset, filepath: Path) -> bool:
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


def exists(_asset: Asset, filepath: Path) -> bool:
    """A :data:`~foehn.transfer.SkipRule` for static assets: fetch each one once."""
    return filepath.exists()


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


def download_indoor_zip(
    dataset: str, bronze_dir: Path, *, force: bool = False, fetcher: Fetcher, **_: object
) -> DownloadResult:
    """Download and extract the indoor climate scenarios ZIP.

    A :class:`~foehn.registry.DownloadAdapter` that does not go through
    :func:`stac_download`: the collection ships a single ``.csv.zip`` rather than
    individual CSV assets, and its skip rule is "has anything been extracted",
    which is a property of the output directory rather than of one asset.
    """
    collection_id = COLLECTIONS[dataset]
    out_dir = bronze_dir / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    if not force and any(out_dir.glob("*.csv")):
        logger.info("  %s already extracted — skipping", dataset)
        return DownloadResult(total_assets=1, downloaded=0, skipped=1, filenames=[])

    items = fetcher.items(collection_id)
    archives = assets_of(items, suffixes=(".zip",))
    archive = archives[0] if archives else None
    if archive is None:
        logger.warning("  No .zip asset found for %s", dataset)
        return DownloadResult()

    _banner(f"Indoor scenarios: {collection_id}", out_dir)

    zip_path = out_dir / archive.name
    fetcher.stream(archive.href, zip_path, timeout=300)
    logger.info("  Downloaded: %s (%.1f MB)", zip_path.name, zip_path.stat().st_size / 1e6)

    extracted = _safe_extract_zip(zip_path, out_dir)
    logger.info("  Extracted %d files", extracted)

    return DownloadResult(total_assets=1, downloaded=1, skipped=0, filenames=[zip_path.name])
