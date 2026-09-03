"""Getting a grid kind's matched files onto disk.

What :mod:`foehn.transfer` and :mod:`foehn.assets` are to the download paths,
for the one path that reads what it fetches: list the collection, keep the
assets a ``match`` selects, refuse a match wider than the kind can hold, fetch
whatever is missing. :mod:`foehn.grids` opens the files this hands back and
knows nothing about where they came from.

Called from :mod:`foehn.registry`, configured by the :class:`~foehn.grids.GridReader`
on the kind's row — the suffixes, the file caps and whether a STAC ``datetime``
means a model run are all facts about the kind, stated there.
"""

from __future__ import annotations

import logging
import re
import warnings
from datetime import UTC, datetime
from pathlib import Path

from foehn.assets import Asset, assets_of, other_extensions
from foehn.atomicwrite import write_text
from foehn.collections import COLLECTIONS
from foehn.fetch import Fetcher, FetchError
from foehn.transfer import already_current, fetch_all
from foehn.workspace import Workspace

logger = logging.getLogger(__name__)


# Grid collections are listed on *every* open, including repeat opens of an
# already-cached file: ensure_grid_files deliberately verifies the match against
# the collection rather than the local cache. That is why this path lists with
# cache=True while the download paths do not — noticing what changed upstream is
# their job. How long a cached listing stays valid is derived from what the walk
# cost, which only the fetcher can measure, so it lives there (see
# foehn.fetch._LISTING_TTL_FACTOR).


# A GRIB2 forecast filename carries its model run as a bare YYYYMMDDHHMM stamp
# (icon-ch1-eps-202605231500-0-t_2m-ctrl.grib2), which is also what a caller's
# ``match`` must contain to select a single field.
_RUN_STAMP_RE = re.compile(r"(?<!\d)(\d{12})(?!\d)")


def _run_datetime_filter(match: str | None) -> str | None:
    """Map a ``match`` onto a STAC ``datetime`` query, when it names one run.

    The forecast collections are one STAC item *per file* — forecast_icon_ch1 is
    ~57,000 items over ~571 pages, about 170s to walk in full, and that walk runs
    on every open. Their item ``datetime`` is the model run time, so a match that
    names a run narrows the listing server-side to that run's ~200 items (~0.5s).

    Only reached for a kind whose :attr:`GridReader.run_datetime` says its
    ``datetime`` means that. The other listings are small anyway (radar_precip is
    16 items), so they lose nothing by walking in full.
    """
    if match is None:
        return None
    found = _RUN_STAMP_RE.search(match)
    if not found:
        return None
    try:
        run = datetime.strptime(found.group(1), "%Y%m%d%H%M").replace(tzinfo=UTC)
    except ValueError:
        return None
    return run.strftime("%Y-%m-%dT%H:%M:%SZ")


# A refresh that fails part-way can leave a set where some files are the new
# generation and the rest the old one. That fact has to outlive the process: the
# next attempt would otherwise measure the mixed cache it inherited, find nothing
# changed since *it* started, and hand it back as complete — and an attempt that
# cannot reach the listing at all has no Asset metadata to judge coherence with.
_INCOHERENT_MARKER = ".foehn-incoherent"


def _fingerprint(path: Path) -> tuple[int, int] | None:
    """What a file looks like right now, or None if it is not there."""
    try:
        info = path.stat()
    except OSError:
        return None
    return (info.st_mtime_ns, info.st_size)


def _mark_incoherent(out_dir: Path, detail: str) -> None:
    """Record that this dataset's cache mixes generations."""
    try:
        write_text(out_dir / _INCOHERENT_MARKER, f"{datetime.now(UTC).isoformat()}\n{detail}\n")
    except OSError:  # pragma: no cover - a cache we cannot write to is one we cannot mark
        logger.debug("Could not write the incoherence marker in %s", out_dir)


def _clear_incoherent(out_dir: Path) -> None:
    """A complete refresh puts every file at the same generation again."""
    (out_dir / _INCOHERENT_MARKER).unlink(missing_ok=True)


def _raise_if_incoherent(dataset: str, out_dir: Path) -> None:
    """Refuse to *hand back* a cache a previous run recorded as mixed.

    Checked where the cache would be returned, never before the fetch: a marker
    that blocked the refresh too would make the mix permanent, since finishing
    the download is exactly what repairs it.
    """
    marker = out_dir / _INCOHERENT_MARKER
    if not marker.exists():
        return
    raise FetchError(
        f"The local cache for {dataset!r} mixes file generations: a previous refresh failed part-way "
        f"through and the set was never completed. It is not safe to read. Re-run with the collection "
        f"reachable to finish the refresh, or delete {out_dir} to start clean."
    )


def _raise_if_too_many(dataset: str, match: str | None, names: list[str], max_files: int | None) -> None:
    """Refuse a match that resolves to more files than the caller can handle at once."""
    if max_files is None or len(names) <= max_files:
        return
    examples = "\n".join(f"  {n}" for n in sorted(names)[:4])
    more = "" if len(names) <= 4 else f"\n  ... and {len(names) - 4} more"
    if max_files == 1:
        detail = "but this collection is read one file at a time. Narrow match= to pick a single file"
    else:
        detail = (
            f"which exceeds the {max_files}-file cap (the whole set is opened at once, and downloading "
            "a collection this size is rarely what was meant). Narrow match= to a smaller set"
        )
    raise ValueError(
        f"match={match!r} matched {len(names)} files for {dataset!r}, {detail}. Matches include:\n{examples}{more}"
    )


def _grid_assets(items: list[dict], suffixes: tuple[str, ...], match: str | None) -> tuple[list[Asset], set[str]]:
    """Pick the grid assets out of a STAC listing.

    Returns the matching assets plus the other extensions seen, which feed the
    "available asset types" hint when nothing matches.
    """
    matched = assets_of(items, suffixes=suffixes, contains=match)
    other_exts = {ext for ext in other_extensions(items) if not ext.endswith(suffixes)}
    return matched, other_exts


def ensure_grid_files(
    dataset: str,
    workspace: Workspace,
    *,
    suffixes: tuple[str, ...] = (".nc",),
    match: str | None = None,
    max_files: int | None = None,
    run_datetime: bool = False,
    fetcher: Fetcher,
) -> list[Path]:
    """Return local grid files for a collection, downloading them if absent.

    Only assets whose name ends in one of ``suffixes`` are fetched (``.nc`` for
    NetCDF, ``.grib2``/``.grib`` for GRIB2); other payloads (GeoTIFF/ZIP copies)
    are ignored. ``match`` keeps only files whose name contains the given
    substring, which is how callers narrow a heterogeneous multi-file collection
    to a coherent set.

    Always consults the remote listing first (when reachable), so it can (a)
    verify a single-file (GRIB2/radar) match resolves to exactly one file in the
    *collection* — not merely one file in a sparse local cache — and (b) avoid
    handing back a partial multi-file cache left by an interrupted download. Only
    files missing or older than their Asset's STAC ``updated`` value are
    downloaded. If the listing is
    unreachable, it falls back to the cache (with a warning), still enforcing the
    per-format file cap on whatever is cached.
    """
    out_dir = workspace.bronze(dataset)
    local = sorted(f for s in suffixes for f in out_dir.glob(f"*{s}"))

    sfx = "/".join(suffixes)
    collection_id = COLLECTIONS[dataset]
    datetime_filter = _run_datetime_filter(match) if run_datetime else None
    try:
        items = fetcher.items(collection_id, cache=True, datetime_filter=datetime_filter)
    except FetchError as exc:
        cached = [f for f in local if match is None or match in f.name]
        if cached:
            # Offline: can't check the collection, but still enforce the cap on
            # what's cached so an over-broad match can't slip through silently.
            # The marker is the only thing that can speak for coherence here —
            # there is no Asset metadata to compare the files against.
            _raise_if_incoherent(dataset, out_dir)
            _raise_if_too_many(dataset, match, [f.name for f in cached], max_files)
            warnings.warn(
                f"Could not reach the STAC API to verify the {dataset!r} cache "
                f"({type(exc).__name__}); using {len(cached)} locally cached file(s) without "
                "checking the collection for a complete/unique match.",
                stacklevel=2,
            )
            return cached
        raise

    matched, other_exts = _grid_assets(items, suffixes, match)

    if not matched and datetime_filter is not None:
        # The run-narrowed listing came back with nothing usable. Rather than
        # report a file as missing on the strength of an optimisation, fall back
        # to the full walk — slow, but it is exactly what happened before.
        logger.debug("No %s assets for %r under datetime=%s; retrying unfiltered", sfx, match, datetime_filter)
        items = fetcher.items(collection_id, cache=True)
        matched, other_exts = _grid_assets(items, suffixes, match)

    if not matched:
        if match is not None:
            raise ValueError(f"No {sfx} assets matching {match!r} found for {dataset!r}.")
        found = ", ".join(sorted(other_exts)) or "none"
        raise ValueError(f"No {sfx} assets found for {dataset!r} (available asset types: {found}).")

    # Enforce the per-format file cap before downloading so an over-broad match
    # (e.g. a whole forecast run) can't pull hundreds of files off the network.
    _raise_if_too_many(dataset, match, [a.name for a in matched], max_files)

    # Recorded before the fetch so the failure path can see what it replaced.
    before = {asset.name: _fingerprint(out_dir / asset.name) for asset in matched}

    # ``on_error="raise"`` rather than the download paths' count-and-continue: a
    # grid read cannot proceed on a partial set, so the first failure is fatal.
    # Destination de-duplication and the worker pool are the transfer module's.
    try:
        fetch_all(
            matched,
            out_dir,
            fetcher=fetcher,
            skip=already_current,
            on_error="raise",
            label="grid file",
        )
    except FetchError as exc:
        cached = [out_dir / asset.name for asset in matched]
        if not all(path.exists() for path in cached):
            raise
        # Presence is not coherence. ``fetch_all`` refreshes assets one at a
        # time, so a failure part-way through can leave some files at the new
        # generation and the rest at the old one — all of them present. Reading
        # that set builds a Dataset that silently mixes generations.
        #
        # Coherence is a property of the cache against the Collection, not of
        # what this run happened to touch. Judging it by what this run replaced
        # only caught the failure on the run that made it: the next attempt
        # re-baselined against the mixed cache it inherited, found nothing
        # changed, and handed it back as complete. A same-size, same-mtime
        # replacement slipped past that check too.
        #
        # So compare each file against its own Asset. Every file current is
        # coherent; every file stale is also coherent — that is the offline
        # cache, one whole generation behind, and the fallback this branch
        # exists for. A mix of the two is the state no reader can use.
        #
        # Two independent signals, because neither covers the other. What this
        # run replaced catches the failure on the run that causes it, whatever
        # the Asset metadata says. Comparing each file against its own Asset
        # catches a set already mixed on arrival, and does not care whether a
        # replacement happened to land on the same size and mtime. Either one
        # means no reader can use this set.
        replaced = [path.name for path in cached if _fingerprint(path) != before.get(path.name)]
        stale = [asset.name for asset in matched if not already_current(asset, out_dir / asset.name)]
        if (0 < len(replaced) < len(cached)) or (0 < len(stale) < len(matched)):
            detail = f"{dataset}: replaced {len(replaced)}/{len(cached)}, {len(stale)} still behind"
            _mark_incoherent(out_dir, detail)
            raise FetchError(
                f"Could not refresh {dataset!r} ({type(exc).__name__}: {exc}) part-way through: "
                f"{len(replaced)} of {len(cached)} file(s) were replaced. A set that mixes generations "
                "is not safe to read; re-run to finish the refresh."
            ) from exc
        # Nothing this run replaced, and every file agrees with its Asset — but
        # an earlier run may have left the mix that this one could not repair.
        _raise_if_incoherent(dataset, out_dir)
        warnings.warn(
            f"Could not refresh {dataset!r} ({type(exc).__name__}); using the complete local cache.",
            stacklevel=2,
        )
        return sorted(cached)
    _clear_incoherent(out_dir)
    return sorted({out_dir / asset.name for asset in matched})


__all__ = ["ensure_grid_files"]
