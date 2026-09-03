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

import json
import logging
import re
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from foehn._locking import exclusive_lock
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
# A refresh that fails part-way can leave a set where some files are the new
# generation and the rest the old one. That fact has to outlive the process: the
# next attempt would otherwise measure the mixed cache it inherited, find nothing
# changed since *it* started, and hand it back as complete — and an attempt that
# cannot reach the listing at all has no Asset metadata to judge coherence with.
#
# Named files, not a dataset-wide flag. One collection holds many independent
# parameter sets, and a completed ``tabs`` refresh says nothing about whether
# ``rhiresd`` was ever finished — clearing the whole dataset on it released
# exactly the files still known to be mixed.
_INCOHERENT_MARKER = ".foehn-incoherent.json"
_REFRESH_LOCK = ".foehn-refresh.lock"


@contextmanager
def _refresh_lock(out_dir: Path) -> Iterator[None]:
    """Hold one dataset's refresh for its whole lifecycle.

    Not just the marker's read-modify-write. Two refreshes of the *same* match
    each checked coherence, each downloaded, and each published into the same
    directory — interleaving their writes into a set neither one would have
    produced, then both clearing the marker on the way out. An offline reader
    could likewise pass the coherence check and return paths a refresh
    republished while it was deciding.

    Serializing a whole refresh means a second caller waits for a download it
    would otherwise have duplicated. That is the cost; the alternative is a
    Dataset assembled from two generations at once.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(out_dir / _REFRESH_LOCK):
        yield


def _read_incoherent(out_dir: Path) -> set[str] | None:
    """Which files a previous run left at an unknown generation.

    None means the marker exists but cannot be understood. That is not the same
    as "nothing pending": a marker truncated by the crash it was recording would
    otherwise fail open and release the very files it was written to protect.
    """
    marker = out_dir / _INCOHERENT_MARKER
    try:
        raw = marker.read_text(encoding="utf-8")
    except FileNotFoundError:
        return set()
    except OSError:
        return None
    try:
        recorded = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(recorded, dict):
        return None
    pending = recorded.get("pending")
    if not isinstance(pending, list):
        return None
    return {str(name) for name in pending}


def _write_incoherent(out_dir: Path, names: set[str]) -> None:
    marker = out_dir / _INCOHERENT_MARKER
    if not names:
        marker.unlink(missing_ok=True)
        return
    write_text(marker, json.dumps({"pending": sorted(names)}, indent=2))


def _mark_incoherent(out_dir: Path, names: list[str]) -> None:
    """Record that these files may be at different generations from each other.

    Written *before* the fetch, not after the failure. A process killed outright
    between publishing one file and the next never reaches an exception handler,
    and the set it leaves behind is exactly the one this is for. The cost is one
    small write per refresh that actually has something to download.

    Callers hold :func:`_refresh_lock`.
    """
    known = _read_incoherent(out_dir)
    if known is None:
        # Unreadable, and an unreadable marker already blocks everything it
        # could name. Merging into it would turn "state unknown" into a tidy
        # empty set that the next successful refresh of *any* set then deletes,
        # releasing the files it was protecting.
        return
    _write_incoherent(out_dir, known | set(names))


def _clear_incoherent(out_dir: Path, names: list[str]) -> None:
    """Release only the files this run actually brought to one generation.

    Callers hold :func:`_refresh_lock`, and reach here only after a refresh that
    completed. A failed refresh never clears: whether it left the set mixed is
    exactly what it cannot answer.
    """
    pending = _read_incoherent(out_dir)
    if pending is None:
        # Unreadable: leave it alone rather than replace it with a guess.
        return
    if pending & set(names):
        _write_incoherent(out_dir, pending - set(names))


def _blocked_names(out_dir: Path, names: list[str]) -> set[str]:
    """The subset of *names* a previous run recorded as being at an unknown generation."""
    pending = _read_incoherent(out_dir)
    if pending is None:
        return set(names)  # unreadable marker blocks everything it could have named
    return pending & set(names)


def _raise_if_incoherent(dataset: str, out_dir: Path, names: list[str]) -> None:
    """Refuse to *hand back* files a previous run recorded as mixed.

    Checked where the cache would be returned, never before the fetch: a marker
    that blocked the refresh too would make the mix permanent, since finishing
    the download is exactly what repairs it.
    """
    blocked = _blocked_names(out_dir, names)
    if not blocked:
        return
    raise FetchError(
        f"The local cache for {dataset!r} mixes file generations: a previous refresh failed part-way "
        f"through and {len(blocked)} of these file(s) were never completed. It is not safe to read. "
        f"Re-run with the collection reachable to finish the refresh, or delete {out_dir} to start clean."
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

    # Everything from here — the coherence check, the listing, the download, the
    # marker, and the decision to hand paths back — happens under one lock. The
    # checks are only worth anything if the set cannot change between making them
    # and returning.
    with _refresh_lock(out_dir):
        return _refresh(
            dataset,
            out_dir,
            suffixes=suffixes,
            match=match,
            max_files=max_files,
            run_datetime=run_datetime,
            fetcher=fetcher,
        )


def _refresh(
    dataset: str,
    out_dir: Path,
    *,
    suffixes: tuple[str, ...],
    match: str | None,
    max_files: int | None,
    run_datetime: bool,
    fetcher: Fetcher,
) -> list[Path]:
    """One dataset's refresh, with :func:`_refresh_lock` held throughout."""
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
            _raise_if_incoherent(dataset, out_dir, [f.name for f in cached])
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

    names = [asset.name for asset in matched]
    cached = [out_dir / name for name in names]

    # Files a previous run left at an unknown generation have to be fetched
    # again whatever the freshness rule says. When a Collection omits ``updated``
    # — or states it in a form that will not parse — every local file counts as
    # current, the refresh skips the lot, and clearing the marker on that
    # "success" released the mix untouched.
    forced = _blocked_names(out_dir, names)
    skip = (
        already_current
        if not forced
        else (lambda asset, path: path.name not in forced and already_current(asset, path))
    )

    # Marked before a byte moves. A process killed outright between publishing
    # one file and the next never reaches an exception handler, so a marker
    # written only on failure was never there when it mattered most.
    refreshing = [asset.name for asset in matched if not skip(asset, out_dir / asset.name)]
    marked = len(matched) > 1 and bool(refreshing)
    if marked:
        _mark_incoherent(out_dir, names)

    # ``on_error="raise"`` rather than the download paths' count-and-continue: a
    # grid read cannot proceed on a partial set, so the first failure is fatal.
    # Destination de-duplication and the worker pool are the transfer module's.
    try:
        fetch_all(
            matched,
            out_dir,
            fetcher=fetcher,
            skip=skip,
            on_error="raise",
            label="grid file",
        )
    except BaseException as exc:
        # A refresh that did not finish cannot say whether it left the set mixed,
        # so it never clears the mark it made. Detecting "nothing was replaced"
        # by size and mtime looked like a safe exception and was not: a file can
        # be rewritten with different bytes at the same length and its mtime put
        # back, and both signals then say nothing happened.
        if not isinstance(exc, FetchError):
            raise  # the mark is on disk; propagate the real cause unchanged
        if marked:
            raise FetchError(
                f"Could not refresh {dataset!r} ({type(exc).__name__}: {exc}). The refresh did not "
                "complete, so this set may now mix file generations and is not safe to read; re-run "
                "to finish it."
            ) from exc
        if not all(path.exists() for path in cached):
            raise

        # Nothing needed refreshing in the first place, so there was never
        # anything to leave half-done — but an earlier run's mix still stands.
        _raise_if_incoherent(dataset, out_dir, names)
        warnings.warn(
            f"Could not refresh {dataset!r} ({type(exc).__name__}); using the complete local cache.",
            stacklevel=2,
        )
        return sorted(cached)

    # Every file in *this* set is now at one generation. Files outside it stay
    # marked: finishing ``tabs`` says nothing about whether ``rhiresd`` ever
    # completed.
    _clear_incoherent(out_dir, names)
    return sorted(set(cached))


__all__ = ["ensure_grid_files"]
