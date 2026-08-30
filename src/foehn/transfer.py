"""Turning a set of :class:`~foehn.assets.Asset` into files on disk.

:mod:`foehn.fetch` owns *one* HTTP call. This module owns *many*: the worker
pool, destination de-duplication, per-asset failure isolation, ETag bookkeeping,
atomic writes and the counting that produces a :class:`DownloadResult`.

Before this module that loop was written five times — once in each of the four
download paths in ``client``, and once more in ``grids.ensure_grid_files``. All five
were the same body; only three things ever varied, and they are the three
parameters :func:`fetch_all` takes:

* which assets to skip without asking the network (``skip``),
* how one asset becomes one file (``write``),
* whether a failed asset is counted or fatal (``on_error``).

The duplication also showed in the interface: ``DownloadResult.total_assets``
used to mean something different depending on which of the five produced it, so
a caller had to know its provenance to read the field. With one producer it has
one meaning, and ``downloaded + skipped + failed == total_assets`` holds.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from foehn.assets import Asset
from foehn.fetch import DEFAULT_WORKERS, Fetcher
from foehn.meteocsv import utf8_meteoswiss_csv

logger = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    """Summary of a download call.

    Callers use this to decide whether to run expensive downstream work
    (e.g. Spark MERGE INTO) without scanning the output directory.

    ``downloaded + skipped + failed == total_assets`` always holds:
    ``total_assets`` is every asset handed to :func:`fetch_all`, ``skipped``
    counts both the ones ``skip`` rejected locally and the ones the server
    answered 304 for, and destination collisions count as skipped too.
    """

    total_assets: int = 0
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    filenames: list[str] = field(default_factory=list)

    def __add__(self, other: DownloadResult) -> DownloadResult:
        """Combine two results, for callers that run more than one pass.

        The standard CSV path downloads collection metadata and item data as two
        passes and reports them as one; both of its callers used to sum the
        fields by hand, and only one of them summed all of them.
        """
        return DownloadResult(
            total_assets=self.total_assets + other.total_assets,
            downloaded=self.downloaded + other.downloaded,
            skipped=self.skipped + other.skipped,
            failed=self.failed + other.failed,
            filenames=self.filenames + other.filenames,
        )


# --- Atomic writes ---


def atomic_write_bytes(path: Path, data: bytes | memoryview) -> None:
    """Write bytes via a sibling temp file + Path.replace so readers never see a torn write."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    """Write text via a sibling temp file + Path.replace so readers never see a torn write."""
    atomic_write_bytes(path, text.encode("utf-8"))


# --- Writers: how one asset becomes one file ---


@dataclass(frozen=True)
class WriteResult:
    """What a :data:`Writer` did with one asset.

    ``downloaded=False`` is the conditional-GET case: the server said 304, so
    the file on disk is already current. ``etag`` is the validator to store for
    next time, or None when the response carried none.
    """

    downloaded: bool
    etag: str | None = None


Writer = Callable[[Fetcher, Asset, Path, str | None], WriteResult]
"""``(fetcher, asset, destination, stored etag) -> WriteResult``. Runs on a worker thread."""

SkipRule = Callable[[Asset, Path], bool]
"""``(asset, destination) -> skip it?``. Runs on the main thread, before anything is queued."""


def exists(_asset: Asset, filepath: Path) -> bool:
    """A :data:`SkipRule` for static assets: fetch each one once.

    Lives here rather than with either caller: it was defined identically in the
    download paths and in the gridded read path, which is one rule and two places
    for it to change.
    """
    return filepath.exists()


def already_current(asset: Asset, filepath: Path) -> bool:
    """A :data:`SkipRule`: is the local copy of *asset* up to date?

    Handles MeteoSwiss's in-place overwrites — e.g. CombiPrecip reanalysis
    (CPCH) replaces the original CPC hourly file with the same filename
    ~8 days later. A plain :func:`exists` check would leave the stale version
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


def stream_to_disk(fetcher: Fetcher, asset: Asset, path: Path, etag: str | None) -> WriteResult:
    """Stream a binary asset straight to disk. Used for GRIB2, HDF5, NetCDF and ZIP."""
    fetcher.stream(asset.href, path)
    return WriteResult(downloaded=True)


def csv_to_disk(fetcher: Fetcher, asset: Asset, path: Path, etag: str | None) -> WriteResult:
    """Fetch a CSV conditionally, re-encode to UTF-8, and write it.

    Only sends the stored ETag when the local file is still there — otherwise a
    304 would report "skipped" over a file that no longer exists.

    MeteoSwiss CSVs are usually UTF-8 but some are Windows-1252; normalise to
    UTF-8. Bytes in, bytes out — decoding to str here only to re-encode on write
    would hold three copies of a multi-hundred-MB CSV at once, times ``workers``.
    """
    validator = etag if etag and path.exists() else None
    fetched = fetcher.get(asset.href, etag=validator, timeout=60)
    if fetched.not_modified:
        return WriteResult(downloaded=False)
    atomic_write_bytes(path, utf8_meteoswiss_csv(fetched.body))
    return WriteResult(downloaded=True, etag=fetched.etag)


# --- The engine ---

OnError = Literal["count", "raise"]


def _dedupe_by_destination(pairs: list[tuple[Asset, Path]]) -> tuple[list[tuple[Asset, Path]], int]:
    """Keep one (asset, destination) pair per destination path, and count the drops.

    Two hrefs resolving to the same local filename would have workers streaming
    into the same ``.part`` file at once and corrupting it. A collision is an
    upstream naming quirk we can't resolve here, so keep the first and say so
    rather than dropping one silently.
    """
    kept: dict[Path, Asset] = {}
    dropped = 0
    for asset, filepath in pairs:
        if filepath in kept:
            logger.warning("  Duplicate destination %s — skipping %s", filepath.name, asset.href)
            dropped += 1
            continue
        kept[filepath] = asset
    return [(asset, filepath) for filepath, asset in kept.items()], dropped


def fetch_all(
    assets: Iterable[Asset],
    out_dir: Path,
    *,
    fetcher: Fetcher,
    workers: int = DEFAULT_WORKERS,
    write: Writer = stream_to_disk,
    skip: SkipRule | None = None,
    etags: dict[str, str] | None = None,
    on_error: OnError = "count",
    label: str = "file",
) -> DownloadResult:
    """Fetch *assets* into *out_dir*, concurrently, and report what happened.

    Args:
        assets: What to fetch. Each lands at ``out_dir / asset.name``.
        out_dir: Destination directory, created if absent.
        fetcher: The network port. Safe to share across the pool — it hands each
            worker thread its own session.
        workers: Concurrent transfers.
        write: How one asset becomes one file. :func:`stream_to_disk` for
            binaries, :func:`csv_to_disk` for the conditional-GET CSV path.
        skip: Decided locally, before anything is queued — an existence check, or
            a "local mtime is newer than the STAC ``updated``" check. Assets it
            rejects are counted as skipped and never touched.
        etags: The ETag store, read when submitting and updated as results
            arrive. Mutated in place; the caller owns loading and saving it.
        on_error: ``"count"`` isolates a failed asset so one transient 404 or
            timeout can't abort the batch (and discard the ETags collected from
            the others). ``"raise"`` re-raises the first failure, for callers
            that cannot proceed with a partial set — a grid read needs its files.
        label: Noun for the log lines ("CSV", "metadata", "binary", "grid").

    Returns:
        A :class:`DownloadResult` whose four counts sum to ``total_assets``.
    """
    assets = list(assets)
    out_dir.mkdir(parents=True, exist_ok=True)

    queued: list[tuple[Asset, Path]] = []
    preskipped = 0
    for asset in assets:
        filepath = out_dir / asset.name
        if skip is not None and skip(asset, filepath):
            preskipped += 1
            continue
        queued.append((asset, filepath))

    queued, collisions = _dedupe_by_destination(queued)
    skipped = preskipped + collisions

    total = len(queued)
    downloaded = 0
    failed = 0
    filenames: list[str] = []

    if total:
        logger.info("  %d %s(s) to process", total, label)
        with ThreadPoolExecutor(max_workers=min(workers, total)) as pool:
            # ETags are read here, on the main thread, and written below as
            # results arrive — also on the main thread. The store is never
            # touched from a worker.
            future_to_asset = {
                pool.submit(write, fetcher, asset, filepath, (etags or {}).get(asset.href)): asset
                for asset, filepath in queued
            }
            for i, fut in enumerate(as_completed(future_to_asset), 1):
                asset = future_to_asset[fut]
                try:
                    outcome = fut.result()
                except Exception as exc:
                    if on_error == "raise":
                        raise
                    failed += 1
                    logger.warning("  [%d/%d] FAILED: %s — %s", i, total, asset.href, exc)
                    continue
                if not outcome.downloaded:
                    skipped += 1
                    continue
                if etags is not None:
                    if outcome.etag:
                        etags[asset.href] = outcome.etag
                    else:
                        # Server returned 200 without an ETag. Keeping the old one
                        # would re-send it as If-None-Match forever, so this asset
                        # would re-download every run and never once be skipped.
                        etags.pop(asset.href, None)
                downloaded += 1
                filenames.append(asset.name)
                logger.info("  [%d/%d] Downloaded: %s", i, total, asset.name)

    if skipped:
        logger.info("  Skipped %d unchanged %s(s)", skipped, label)
    if failed:
        logger.warning("  %d %s(s) failed to download", failed, label)
    logger.info("  Done — %d %s(s) downloaded", downloaded, label)

    return DownloadResult(
        total_assets=len(assets),
        downloaded=downloaded,
        skipped=skipped,
        failed=failed,
        filenames=filenames,
    )


__all__ = [
    "DownloadResult",
    "SkipRule",
    "WriteResult",
    "Writer",
    "already_current",
    "atomic_write_bytes",
    "atomic_write_text",
    "csv_to_disk",
    "exists",
    "fetch_all",
    "stream_to_disk",
]
