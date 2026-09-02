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
from pathlib import Path

from foehn.assets import Asset, assets_of, collection_assets, latest_run_of, select
from foehn.collections import CLIMATE_NORMALS_ZIP_URL, COLLECTIONS
from foehn.fetch import DEFAULT_WORKERS, Fetcher
from foehn.state import EtagRun
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
        workspace: Workspace,
        *,
        time_slice: list[str],
        since: str | None = None,
        workers: int = DEFAULT_WORKERS,
        fetcher: Fetcher,
        **_: object,
    ) -> DownloadResult:
        result = DownloadResult()
        if with_metadata:
            result = download_metadata(dataset, workspace, workers=workers, fetcher=fetcher)

        collection_id = COLLECTIONS[dataset]
        out_dir = workspace.bronze(dataset)
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
        etag_run = EtagRun.begin(workspace) if etags else None
        store = etag_run.values if etag_run is not None else None
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

        if etag_run is not None:
            # Only on a clean full listing: with ``since`` the item list is
            # partial, and after failures the universe may be incomplete.
            listed = {a.href for a in every} if since is None and fetched.failed == 0 else None
            etag_run.commit(collection_id, listed=listed)

        return result + fetched

    return download


# --- Metadata downloads ---


def download_metadata(
    dataset: str, workspace: Workspace, workers: int = DEFAULT_WORKERS, *, fetcher: Fetcher
) -> DownloadResult:
    """Download collection-level metadata files (stations, parameters, inventory)."""
    coll = fetcher.collection(COLLECTIONS[dataset])
    return fetch_all(
        collection_assets(coll, suffixes=(".csv",)),
        workspace.bronze(dataset),
        fetcher=fetcher,
        workers=workers,
        write=csv_to_disk,
        label="metadata file",
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

    if not force and materialization_current(archive, out_dir):
        logger.info("  %s already extracted and current — skipping", dataset)
        return DownloadResult(total_assets=1, downloaded=0, skipped=1, filenames=[])

    _banner(f"Indoor scenarios: {collection_id}", out_dir)
    return materialize_archive(archive, out_dir, fetcher=fetcher, force=force, timeout=300)
