"""How one dataset's assets become files in :term:`Bronze`.

One :func:`stac_download` engine configured per dataset kind, plus the two kinds
that do not list STAC at all and ship a ZIP instead. Every adapter here has the
shape :class:`~foehn.registry.DownloadAdapter` names, so the registry row holds
one whichever path a kind takes.

Was ``client`` — a name :file:`CONTEXT.md` tells you not to use (the **Fetcher**
entry ends "_Avoid_: client, session, transport"), on the one module in the tree
that makes no HTTP call itself. It also held the state files and the ZIP guards,
which are :mod:`foehn.state` and :mod:`foehn.archives` now.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

from foehn import coherence
from foehn.assets import Asset, assets_of, collection_assets, latest_run_of, select
from foehn.collections import (
    CLIMATE_NORMALS_ZIP_URL,
    COLLECTIONS,
    ROLLING_TIME_SLICES,
    forecast_run_from_filename,
    time_slice_from_filename,
)
from foehn.fetch import DEFAULT_WORKERS, Fetcher
from foehn.retention import record_listing
from foehn.state import EtagRun, parse_instant
from foehn.transfer import (
    DownloadResult,
    SkipRule,
    Writer,
    csv_to_disk,
    fetch_all,
    materialization_current,
    materialize_archive,
    stream_to_disk,
)
from foehn.workspace import Workspace

logger = logging.getLogger(__name__)


# --- One listing path, configured per dataset kind ---


def _banner(title: str, out_dir: Path) -> None:
    """The four download paths each printed this by hand, ten lines in total."""
    logger.info("%s", "=" * 60)
    logger.info("%s", title)
    logger.info("Output dir: %s", out_dir)
    logger.info("%s", "=" * 60)


def _since_instant(since: str | None) -> datetime | None:
    """Parse *since* up front, so a malformed one fails before any listing."""
    if not since:
        return None
    cutoff = parse_instant(since)
    if cutoff is None:
        raise ValueError(f"since={since!r} is not an ISO 8601 timestamp with a timezone.")
    return cutoff


def _updated_since(items: list[dict], since: str | None) -> list[dict]:
    """Keep only the items MeteoSwiss restated after *since*.

    Written out verbatim in three of the four download paths, along with its two
    log lines. What ``since`` means is one rule. Compared as instants: as strings,
    ``10:30:00Z`` sorted before ``11:00:00+01:00`` and a change half an hour
    after the cursor was skipped. An Item whose ``updated`` cannot be read is
    kept — a redundant fetch is cheaper than a silent skip.
    """
    cutoff = _since_instant(since)
    if cutoff is None:
        return items

    def changed(item: dict) -> bool:
        updated = parse_instant(item.get("properties", {}).get("updated", ""))
        return updated is None or updated > cutoff

    kept = [item for item in items if changed(item)]
    logger.info("  %d items updated since last run", len(kept))
    return kept


# --- What Bronze keeps ---

Pruner = Callable[[Path, list[Asset]], list[str]]
"""``(dataset's Bronze directory, the Collection's full listing) -> names deleted``.

Called only after a clean run, with the Dataset's lock held, so nothing is
reading or replacing the files it removes. Without one, Bronze only ever grew:
a forecast run is ~1 GB, and a station MeteoSwiss stopped publishing kept its
rolling files in every later table.
"""


def _delete(paths: list[Path]) -> list[str]:
    for path in paths:
        path.unlink(missing_ok=True)
    return [path.name for path in paths]


def prune_unlisted_rolling(out_dir: Path, listed: list[Asset]) -> list[str]:
    """Delete the rolling-slice CSVs (``recent``, ``now``) the Collection no longer lists.

    Only those: a ``historical`` or unsliced file is still a correct record of
    what was measured, and nothing upstream will ever serve it again.
    """
    if not listed:
        # An empty listing is a broken one, not a Collection to empty Bronze for.
        return []
    names = {asset.name for asset in listed}
    record_listing(out_dir, sorted(names))
    return _delete(
        [
            path
            for path in sorted(out_dir.glob("*.csv"))
            if path.name not in names and time_slice_from_filename(path.name) in ROLLING_TIME_SLICES
        ]
    )


def prune_superseded_runs(out_dir: Path, listed: list[Asset]) -> list[str]:
    """Delete Forecast runs older than the one this download selected.

    That run is the newest complete one in the full listing, and a clean run has
    just put all of it in Bronze — so it is what conversion and ingestion read,
    and nothing reads an older one. Anchored on the listing rather than on what
    Bronze holds: on disk, a couple of partial runs left by failed downloads can
    outnumber the complete one and pass for "typical", and deleting by that
    measure would delete the run that is actually whole. A partial run *newer*
    than the selection is kept: it is what the next download finishes.
    """
    keep_from = latest_run_of(listed)
    if keep_from is None:
        return []
    return _delete(
        [
            path
            for path in sorted(out_dir.glob("*.csv"))
            if (run := forecast_run_from_filename(path.name)) is not None and run < keep_from
        ]
    )


def _prune(prune: Pruner | None, out_dir: Path, listed: list[Asset]) -> None:
    if prune is None or not out_dir.is_dir():
        return
    if removed := prune(out_dir, listed):
        logger.info("  Removed %d file(s) no longer published, or replaced by a newer run", len(removed))


def _skip_except(skip: SkipRule | None, forced: set[str]) -> SkipRule:
    """*skip*, but never for a file whose generation is in question."""

    def decide(asset: Asset, path: Path) -> bool:
        if path.name in forced:
            return False
        return bool(skip(asset, path)) if skip is not None else False

    return decide


def _after_cursor(
    wanted: list[Asset], items: list[dict], since: str | None, out_dir: Path, pending: set[str]
) -> list[Asset]:
    """Narrow an incremental run without dropping missing files or pending repairs.

    Called with the Dataset lock held: a forced refresh can invalidate a file
    even when its upstream timestamp is older than the committed cursor.
    """
    if not since:
        return wanted
    changed = {item.get("id", "") for item in _updated_since(items, since)}
    return [
        asset
        for asset in wanted
        if asset.item_id in changed or asset.name in pending or not (out_dir / asset.name).exists()
    ]


def _mark_refresh(
    out_dir: Path,
    wanted: list[Asset],
    skip: SkipRule | None,
    pending: set[str],
    *,
    coherent: bool,
    force: bool,
) -> None:
    """Persist set replacement state, or individual explicitly invalidated CSVs."""
    if coherent:
        replacing = [asset.name for asset in wanted if skip is None or not skip(asset, out_dir / asset.name)]
        if len(wanted) > 1 or force or pending:
            coherence.mark(out_dir, replacing)
    elif force or pending:
        coherence.mark(out_dir, [asset.name for asset in wanted if force or asset.name in pending])


def stac_download(
    *,
    suffixes: tuple[str, ...],
    title: str,
    label: str = "file",
    write: Writer = stream_to_disk,
    skip: SkipRule | None = None,
    etags: bool = False,
    list_items: Callable[[str, Fetcher], list[dict]] | None = None,
    excludes: str | None = None,
    time_sliced: bool = False,
    latest_run: bool = False,
    with_metadata: bool = False,
    coherent: bool = False,
    prune: Pruner | None = None,
) -> Callable[..., DownloadResult]:
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
        skip: Decided locally before anything is queued. A ``force`` call
            bypasses it, and the ETag store too.
        etags: Whether this kind keeps an ETag store (the conditional-GET path).
        list_items: How to list the Collection, for a kind whose plain walk is
            too slow — the GRIB2 forecasts are one item per file. Defaults to
            :meth:`~foehn.fetch.Fetcher.items`. Not a ``max_items`` shortcut:
            the API lists oldest first, so a truncated walk only ever reached the
            oldest run, whose files were already expiring.
        excludes: Drop assets whose name contains this, e.g. the ensemble
            members of a forecast run.
        time_sliced: Whether ``time_slice`` narrows this kind's assets. False
            where the filenames carry no slice.
        latest_run: Keep only the newest **Forecast run**. A forecast_local run
            is ~32 CSVs at ~30 MB; an ICON-CH1 run's control member is ~3,500
            GRIB2 files at ~2 MB.
        with_metadata: Whether the collection-level metadata files are fetched
            in the same pass and reported as one result.
        coherent: Whether this kind's files are read as a *set*, so a half-done
            replacement of one member spoils the rest. True for the grid kinds,
            whose readers open several files as one Dataset; false for the CSV
            kinds, which are converted before anything reads them.
        prune: What to delete from Bronze after a clean run, given the full
            listing. None for the grid kinds: their Bronze is also the
            ``open_dataset`` cache, and for radar it is the archive itself.
    """

    def download(
        dataset: str,
        workspace: Workspace,
        *,
        time_slice: list[str],
        since: str | None = None,
        workers: int = DEFAULT_WORKERS,
        force: bool = False,
        fetcher: Fetcher,
    ) -> DownloadResult:
        _since_instant(since)
        collection_id = COLLECTIONS[dataset]
        out_dir = workspace.bronze(dataset)
        # Listed here rather than inside the metadata pass: the data pass's
        # ETag prune below has to know these hrefs, or it drops their ETags as
        # "no longer listed" and every run downloads the metadata in full.
        metadata = metadata_assets(dataset, fetcher=fetcher) if with_metadata else []
        result = (
            _download_metadata_assets(metadata, dataset, workspace, workers=workers, force=force, fetcher=fetcher)
            if metadata
            else DownloadResult()
        )
        _banner(f"{title}: {collection_id}", out_dir)

        items = fetcher.items(collection_id) if list_items is None else list_items(collection_id, fetcher)
        logger.info("  Found %d items", len(items))

        # ``every`` is the full pre-filter set — the prune universe below, so
        # ETags for slices outside this run's time_slice are kept.
        every = assets_of(items, suffixes=suffixes, excludes=excludes)
        wanted = select(
            every,
            time_slices=time_slice if time_sliced else None,
            latest_run=latest_run,
        )
        if latest_run and (run := latest_run_of(every)) is not None:
            available = len({a.forecast_run for a in every if a.forecast_run is not None})
            logger.info("  Newest complete forecast run: %s (of %d listed)", run, available)

        # The same coherence protocol the read path speaks. This writes into the
        # very directory a grid read opens from, and replacing several files one
        # at a time has a middle — so a download running beside a read used to
        # produce a mixed set with nothing on disk to say so. Grid kinds only:
        # the CSV kinds are converted before anything reads them, and there is no
        # set whose members have to agree.
        #
        # The ETag kinds hold the same per-Dataset lock from the ETag snapshot to
        # its commit. Run state merges were locked but publication was not: an
        # older response could replace a newer file and the newer run then commit
        # its ETag, so every later refresh got a 304 and kept the stale bytes.
        # Per Dataset, so unrelated Datasets still download side by side.
        with coherence.refresh_lock(out_dir) if coherent or etags else nullcontext():
            # Retry state precedes the cursor filter. Otherwise an invalidated
            # file that still exists looks unchanged forever after a failed
            # forced refresh, even though its ETag was correctly discarded.
            forced = coherence.blocked(out_dir, [asset.name for asset in wanted]) if coherent or etags else set()
            wanted = _after_cursor(wanted, items, since, out_dir, forced)
            if since and not wanted:
                logger.info("  Nothing changed — skipping")
                if result.failed == 0:
                    _prune(prune, out_dir, every)
                return result
            names = [asset.name for asset in wanted]
            # The conditional-GET writer decides "unchanged" from the server's 304,
            # so the ETag store is this kind's skip rule.
            etag_run = EtagRun.begin(workspace) if etags else None
            store = etag_run.values if etag_run is not None else None
            if store is not None:
                # Unconditional GETs: a stored ETag would come back 304 and keep
                # whatever is on disk, however damaged. Each asset that downloads
                # records its new ETag again; one that fails stays without one.
                for asset in wanted:
                    if force or asset.name in forced:
                        store.pop(asset.href, None)

            # Files a previous run left half-replaced are fetched again whatever
            # the freshness rule says. Without this a retry found both current —
            # the collection states no ``updated``, or the half-written file's
            # mtime is newer than it — skipped everything, called that success
            # and cleared a marker over a cache it had not repaired.
            if force:
                run_skip = None
            else:
                run_skip = _skip_except(skip, forced) if forced else skip

            # Only the files this run replaces can leave the set mixed; the rest
            # are not touched and keep the generation they had. Marking the whole
            # set made one failed file cost a re-download of every other one on
            # the next run — for radar, a whole day of composites over one 404.
            _mark_refresh(out_dir, wanted, run_skip, forced, coherent=coherent, force=force)
            fetched = fetch_all(
                wanted,
                out_dir,
                fetcher=fetcher,
                workers=workers,
                write=write,
                skip=run_skip,
                etags=store,
                label=label,
            )
            # Only a run that finished can say the set is whole again.
            if coherent and fetched.failed == 0:
                coherence.clear(out_dir, names)
            elif etags:
                # CSV repairs are independent. Keep only unsuccessful repairs
                # pending, so one failed station does not refetch its siblings.
                coherence.clear(out_dir, fetched.filenames)
            # Only a clean run, too: after a failure the partial run on disk may
            # be the newest thing there is, and nothing is lost by waiting.
            if result.failed == fetched.failed == 0:
                _prune(prune, out_dir, every)

            if etag_run is not None:
                # Only on a clean full listing: with ``since`` the item list is
                # partial, and after failures the universe may be incomplete.
                listed = {a.href for a in [*every, *metadata]} if since is None and fetched.failed == 0 else None
                etag_run.commit(collection_id, listed=listed)

        return result + fetched

    return download


# --- Metadata downloads ---


def metadata_assets(dataset: str, *, fetcher: Fetcher) -> list[Asset]:
    """The Collection's metadata files (stations, parameters, inventory)."""
    return collection_assets(fetcher.collection(COLLECTIONS[dataset]), suffixes=(".csv",))


def _download_metadata_assets(
    assets: list[Asset],
    dataset: str,
    workspace: Workspace,
    *,
    workers: int,
    force: bool,
    fetcher: Fetcher,
) -> DownloadResult:
    """Fetch metadata files conditionally, as the data files are.

    They were fetched in full on every run, with no ETag sent — the inventory
    alone is megabytes — and rewritten whether or not anything had changed.
    The ETag transition is this pass's own and prunes nothing: what is still
    listed is the data pass's to decide.
    """
    out_dir = workspace.bronze(dataset)
    with coherence.refresh_lock(out_dir):
        etag_run = EtagRun.begin(workspace)
        if force:
            for asset in assets:
                etag_run.values.pop(asset.href, None)
        result = fetch_all(
            assets,
            out_dir,
            fetcher=fetcher,
            workers=workers,
            write=csv_to_disk,
            etags=etag_run.values,
            label="metadata file",
        )
        etag_run.commit(COLLECTIONS[dataset])
    return result


def download_metadata(
    dataset: str, workspace: Workspace, workers: int = DEFAULT_WORKERS, *, fetcher: Fetcher, force: bool = False
) -> DownloadResult:
    """Download collection-level metadata files (stations, parameters, inventory)."""
    return _download_metadata_assets(
        metadata_assets(dataset, fetcher=fetcher), dataset, workspace, workers=workers, force=force, fetcher=fetcher
    )


# --- C6 climate normals ZIP ---


def download_normals_zip(
    dataset: str, workspace: Workspace, *, force: bool = False, fetcher: Fetcher, **_: object
) -> DownloadResult:
    """Download the C6 climate normals ZIP and extract it.

    A :class:`~foehn.registry.DownloadAdapter` that never lists STAC: this is the
    one dataset MeteoSwiss publishes as a fixed-URL ZIP rather than a collection,
    which is why it used to be a special case in the CLI, in the to-parquet
    command and in the Databricks ingest script instead of a row.
    """
    out_dir = workspace.bronze(dataset)
    asset = Asset.from_stac("normals", {"href": CLIMATE_NORMALS_ZIP_URL})
    # The Dataset's lock, which a conversion holds while it reads: publication
    # replaces the whole directory it would be reading from.
    with coherence.refresh_lock(out_dir):
        if not force and materialization_current(asset, out_dir):
            logger.info("  %s already downloaded and extracted — skipping", dataset)
            return DownloadResult(total_assets=1, downloaded=0, skipped=1, filenames=[])

        _banner("Climate normals (C6): fixed-URL ZIP", out_dir)
        return materialize_archive(asset, out_dir, fetcher=fetcher, force=force, timeout=120)


# --- Indoor climate scenarios ZIP (single .csv.zip of per-station CSVs) ---


def download_indoor_zip(
    dataset: str, workspace: Workspace, *, force: bool = False, fetcher: Fetcher, **_: object
) -> DownloadResult:
    """Download and extract the indoor climate scenarios ZIP.

    A :class:`~foehn.registry.DownloadAdapter` that does not go through
    :func:`stac_download`: the collection ships a single ``.csv.zip`` rather than
    individual CSV assets, and its skip rule is "has anything been extracted",
    which is a property of the output directory rather than of one asset.
    """
    collection_id = COLLECTIONS[dataset]
    out_dir = workspace.bronze(dataset)

    items = fetcher.items(collection_id)
    archives = assets_of(items, suffixes=(".zip",))
    archive = archives[0] if archives else None
    if archive is None:
        logger.warning("  No .zip asset found for %s", dataset)
        return DownloadResult()

    # Held for the same reason as download_normals_zip's.
    with coherence.refresh_lock(out_dir):
        if not force and materialization_current(archive, out_dir):
            logger.info("  %s already extracted and current — skipping", dataset)
            return DownloadResult(total_assets=1, downloaded=0, skipped=1, filenames=[])

        _banner(f"Indoor scenarios: {collection_id}", out_dir)
        return materialize_archive(archive, out_dir, fetcher=fetcher, force=force, timeout=300)
