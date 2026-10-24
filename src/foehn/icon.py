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

import contextlib
import logging
import warnings
from collections.abc import Generator
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


class _DropUnstructuredNotice(logging.Filter):
    """Drop cfgrib's notice that an ICON field carries no lat/lon.

    True of every ICON/KENDA field, and foehn attaches the coordinates from the
    constants file straight after — so the warning tells the caller the one
    thing that is not true of what they get back.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith("ecCodes provides no latitudes/longitudes")


@contextlib.contextmanager
def quiet_cfgrib() -> Generator[None]:
    """Silence the noise cfgrib makes reading an ICON/KENDA GRIB, and nothing else.

    Two sources, both cfgrib's rather than the caller's: the lat/lon notice
    above, and xarray's FutureWarning about ``merge``'s ``compat`` default, which
    cfgrib's own ``open_datasets`` triggers once per variable group.
    """
    notice = _DropUnstructuredNotice()
    cfgrib_log = logging.getLogger("cfgrib.dataset")
    cfgrib_log.addFilter(notice)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="In a future version of xarray the default value for compat",
                category=FutureWarning,
            )
            yield
    finally:
        cfgrib_log.removeFilter(notice)


def unstructured_lonlat(dataset: str, workspace: Workspace, *, fetcher: Fetcher):
    """Return (lat, lon) cell-centre arrays for an ICON/KENDA grid, or (None, None).

    Read from the collection's horizontal-constants GRIB (``tlat``/``tlon`` on
    the same ``values`` dimension as the forecast fields). Cached per collection.
    """
    cache_key = (dataset, str(workspace.root))
    if cache_key in _COORDS_CACHE:
        return _COORDS_CACHE[cache_key]

    lat = lon = None
    path = constants_file(dataset, workspace, fetcher=fetcher)
    if path is not None:
        # Only once there is a file to read: with no constants there is nothing
        # for cfgrib to do, and the grid is returned without coordinates.
        import cfgrib

        with quiet_cfgrib():
            constants = cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})
        for ds in constants:
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


__all__ = ["attach_lonlat", "clear_cache", "constants_file", "quiet_cfgrib", "unstructured_lonlat"]
