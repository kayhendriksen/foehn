"""STAC assets, with the facts their filenames carry already parsed out.

The listing comes back from :mod:`foehn.fetch` as raw STAC dicts, because what
a time slice or a forecast run *is* belongs to MeteoSwiss, not to the transport.
Turning those dicts into :class:`Asset` values, and picking the ones a call
wants, happens here — in one place rather than in each caller.

Before this module, every download and load path walked ``item["assets"]``
itself, stripped the query string itself, and re-derived the slice and run from
the filename itself. The forecast-run rule in particular — collect the runs,
take ``max()``, keep only that one — was written out twice, in
``client.download_collection`` and in ``api.load``, and had to stay in step for
the two to agree on which forecast you get.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from foehn._urls import asset_filename
from foehn.collections import (
    forecast_run_from_filename,
    granularity_from_filename,
    time_slice_from_filename,
)


@dataclass(frozen=True)
class Asset:
    """One downloadable file in a STAC listing.

    ``href`` keeps its query string, because that is what has to be fetched;
    every other field is derived from the query-stripped name, because that is
    what carries meaning. Callers that used to hold a bare href string had to
    remember which of the two they were holding at each step.
    """

    href: str
    name: str
    key: str
    """The asset's key in the item, which is how collection-level assets are found."""

    updated: str
    """Per-asset ``updated`` when the API gives one, else the item's."""

    item_id: str
    time_slice: str | None
    granularity: str | None
    forecast_run: str | None

    @classmethod
    def from_stac(cls, key: str, asset: dict, *, item_id: str = "", item_updated: str = "") -> Asset:
        href = asset.get("href", "")
        name = asset_filename(href)
        return cls(
            href=href,
            name=name,
            key=key,
            # STAC allows a per-asset updated; fall back to the item's.
            updated=asset.get("updated") or item_updated,
            item_id=item_id,
            time_slice=time_slice_from_filename(name),
            granularity=granularity_from_filename(name),
            forecast_run=forecast_run_from_filename(name),
        )

    @property
    def extension(self) -> str:
        """Lowercased suffix including the dot, or "" if the name carries none."""
        _, dot, ext = self.name.rpartition(".")
        return f".{ext.lower()}" if dot else ""


def assets_of(
    items: Iterable[dict],
    *,
    suffixes: tuple[str, ...] | None = None,
    contains: str | None = None,
    excludes: str | None = None,
) -> list[Asset]:
    """Every asset across *items*, narrowed by name.

    ``suffixes`` keeps only those file types, ``contains`` only names holding
    that substring, ``excludes`` drops names holding it (``_meta_``, mostly).
    All three match the query-stripped name, which is the bug this replaces:
    ``href.endswith(".csv")`` silently drops an asset served with a token.
    """
    found: list[Asset] = []
    for item in items:
        item_id = item.get("id", "")
        item_updated = item.get("properties", {}).get("updated", "")
        for key, info in item.get("assets", {}).items():
            asset = Asset.from_stac(key, info, item_id=item_id, item_updated=item_updated)
            if suffixes is not None and not asset.name.endswith(suffixes):
                continue
            if contains is not None and contains not in asset.name:
                continue
            if excludes is not None and excludes in asset.name:
                continue
            found.append(asset)
    return found


def collection_assets(
    collection: dict,
    *,
    suffixes: tuple[str, ...] | None = None,
    contains: str | None = None,
    key_contains: str | None = None,
) -> list[Asset]:
    """Assets hanging off the collection itself rather than off its items.

    These are the metadata files — parameters, stations, inventory — and the
    GRIB2 constants. ``key_contains`` matches the asset's key rather than its
    name, which is how the constants file is identified.
    """
    found = assets_of([collection], suffixes=suffixes, contains=contains)
    if key_contains is not None:
        found = [a for a in found if key_contains in a.key.lower()]
    return found


def select(
    assets: Iterable[Asset],
    *,
    time_slices: Iterable[str] | None = None,
    granularities: Iterable[str] | None = None,
    latest_run: bool = False,
) -> list[Asset]:
    """Narrow *assets* to what a call actually wants.

    ``time_slices`` keeps assets in those slices, and keeps assets that have no
    slice at all — a file with no slice segment is unsliced data that every
    query includes, not data that matches nothing.

    ``latest_run`` keeps only the newest forecast run. A forecast item is one
    *day* holding that day's runs, so picking the newest item is not the same
    thing and picks an empty one: the newest day is created at ~04:00 UTC and
    filled as its runs publish. Runs are zero-padded ``YYYYMMDDHHMM``, so the
    newest is ``max()`` without parsing a datetime. One run is ~32 files at
    ~30 MB; the retained window is ~40 runs.
    """
    found = list(assets)

    if time_slices is not None:
        wanted = set(time_slices)
        found = [a for a in found if a.time_slice is None or a.time_slice in wanted]

    if granularities is not None:
        wanted = set(granularities)
        found = [a for a in found if a.granularity in wanted]

    if latest_run:
        runs = {a.forecast_run for a in found if a.forecast_run is not None}
        if runs:
            newest = max(runs)
            found = [a for a in found if a.forecast_run == newest]

    return found


def latest_run_of(assets: Iterable[Asset]) -> str | None:
    """The newest forecast run present, for callers that want to log it."""
    runs = {a.forecast_run for a in assets if a.forecast_run is not None}
    return max(runs) if runs else None


def hrefs(assets: Iterable[Asset]) -> list[str]:
    """Just the fetchable hrefs, for callers that need nothing else."""
    return [a.href for a in assets]


def other_extensions(items: Iterable[dict]) -> set[str]:
    """Extensions present in *items* — for telling a caller what a collection does hold."""
    return {ext for a in assets_of(items) if (ext := a.extension)}


__all__ = [
    "Asset",
    "assets_of",
    "collection_assets",
    "hrefs",
    "latest_run_of",
    "other_extensions",
    "select",
]
