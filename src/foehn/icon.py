"""ICON's unstructured grid, and where its coordinates come from.

ICON/KENDA GRIB2 fields arrive on a 1-D ``values`` dimension with no lat/lon in
the file; the cell centres live in a separate horizontal-constants asset on the
collection. Joining the two is upstream's convention, so it sits below the grid
reader that needs it — :mod:`foehn.odim`'s counterpart for the GRIB2 kind.

The parsed coordinates are memoised for the life of the process, which is the
one piece of state in the grid path. :func:`clear_cache` is part of the
interface for that reason: the test suite has to reset it between tests, and
reaching into a module global to do that is not an interface.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from foehn.assets import collection_assets
from foehn.collections import COLLECTIONS
from foehn.fetch import Fetcher, FetchError
from foehn.transfer import already_current, fetch_all
from foehn.workspace import Workspace

# Parsed ICON/KENDA cell lat/lon — the constants GRIB is ~11 MB and the same grid
# for every field in a collection, so parse it once. Keyed by (dataset,
# workspace root): the constants file is resolved inside the workspace, so
# keying on the dataset alone hands a second workspace the first one's
# coordinates.
_COORDS_CACHE: dict[tuple[str, str], tuple] = {}


def constants_file(dataset: str, workspace: Workspace, *, fetcher: Fetcher) -> Path | None:
    """Locate (or download) a GRIB2 collection's horizontal-constants file.

    Returns the local path, or None if the collection exposes no such asset.
    The constants file is a collection-level STAC asset (not a per-item one).
    """
    out_dir = workspace.bronze(dataset)
    cached = sorted(out_dir.glob("horizontal_constants*.grib2"))
    try:
        meta = fetcher.collection(COLLECTIONS[dataset])
    except FetchError:
        if cached:
            return cached[0]
        raise
    constants = collection_assets(meta, key_contains="horizontal_constants")
    if not constants:
        return cached[0] if cached else None

    out_dir.mkdir(parents=True, exist_ok=True)
    asset = constants[0]
    path = out_dir / asset.name
    try:
        fetch_all(
            [asset],
            out_dir,
            fetcher=fetcher,
            skip=already_current,
            on_error="raise",
            label="ICON constants",
        )
    except FetchError as exc:
        if cached:
            warnings.warn(
                f"Could not refresh ICON constants for {dataset!r} ({type(exc).__name__}); using the local cache.",
                stacklevel=2,
            )
            return cached[0]
        raise
    return path


def unstructured_lonlat(dataset: str, workspace: Workspace, *, fetcher: Fetcher):
    """Return (lat, lon) cell-centre arrays for an ICON/KENDA grid, or (None, None).

    Read from the collection's horizontal-constants GRIB (``tlat``/``tlon`` on
    the same ``values`` dimension as the forecast fields). Cached per collection.
    """
    cache_key = (dataset, str(workspace.root))
    if cache_key in _COORDS_CACHE:
        return _COORDS_CACHE[cache_key]

    import cfgrib

    lat = lon = None
    path = constants_file(dataset, workspace, fetcher=fetcher)
    if path is not None:
        for ds in cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""}):
            if "tlat" in ds.variables:
                lat = ds["tlat"].values
            if "tlon" in ds.variables:
                lon = ds["tlon"].values
    # Only memoise a real answer. Caching the (None, None) from a transient
    # failure — offline, a half-written constants file — would poison every
    # later open in this process with an ungeoreferenced grid.
    if lat is not None and lon is not None:
        _COORDS_CACHE[cache_key] = (lat, lon)
    return lat, lon


def attach_lonlat(ds, dataset: str, workspace: Workspace, *, fetcher: Fetcher, what: str):
    """Attach ICON cell lat/lon to an unstructured Dataset, best-effort.

    ICON/KENDA are on an unstructured grid (1-D ``values``, no lat/lon in the
    GRIB), so the coordinates come from the collection's constants file. Warn and
    continue if it is unavailable (e.g. offline) rather than failing an
    otherwise-good read.
    """
    if "values" not in ds.dims or "lat" in ds.coords:
        return ds
    try:
        lat, lon = unstructured_lonlat(dataset, workspace, fetcher=fetcher)
        if lat is not None and lon is not None and lat.size == ds.sizes["values"]:
            return ds.assign_coords(
                lat=("values", lat, {"units": "degrees_north", "standard_name": "latitude"}),
                lon=("values", lon, {"units": "degrees_east", "standard_name": "longitude"}),
            )
    except Exception as exc:
        warnings.warn(
            f"Could not attach ICON lat/lon for {dataset!r} ({type(exc).__name__}); {what}",
            stacklevel=2,
        )
    return ds


def clear_cache() -> None:
    """Forget every memoised grid. For tests, which must not inherit each other's."""
    _COORDS_CACHE.clear()


__all__ = ["attach_lonlat", "clear_cache", "constants_file", "unstructured_lonlat"]
