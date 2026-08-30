"""How one gridded :class:`~foehn.collections.DatasetKind` becomes an xarray Dataset.

The grid counterpart to :mod:`foehn.readers`, and placed the same way: below
:mod:`foehn.registry`, so :class:`~foehn.registry.KindSpec` can carry a
:class:`GridReader` beside its ``download``, ``convert`` and ``load`` adapters.
The public ``open_dataset`` and ``to_zarr`` live in :mod:`foehn.api`, exactly as
``load`` does.

This module used to sit *above* the registry and keep a second table of its own,
keyed by the same ``DatasetKind`` — and test that enum by hand eight more times
on top of it: which optional import to require, which reader opens a file,
whether the STAC ``datetime`` is a model run, whether to attach ICON
coordinates, which cube builder ``stack=`` meant, and twice more inside the cube
builders re-checking the kind they had just been routed to. All of it is a row
now, and the routing is the registry's, as it already was for the other three
pipeline stages.

Supported formats
-----------------
* NetCDF (climate grids, normals, scenarios) — xarray auto-detects the engine.
* GRIB2 (ICON-CH1/CH2 forecasts, KENDA analysis) — cfgrib engine.
* HDF5/ODIM radar composites (CombiPrecip, hail) — bespoke ODIM reader. These are
  Cartesian ``COMP`` grids (not polar volumes, so not xradar): read via h5py with
  ODIM gain/offset scaling onto Swiss LV95 x/y coordinates.

All formats install via the single optional 'grids' extra (``pip install
"foehn[grids]"``). GRIB2 and radar collections hold thousands of single-field
files, so opening one requires a ``match`` that resolves to a single file (the
cap is enforced before downloading). ICON GRIB2 comes back on a 1-D ``values``
dimension; cell ``lat``/``lon`` are joined from the collection's horizontal-
constants file (best-effort), so the unstructured grid is geo-referenced.

Opening reads a single file; a kind's :attr:`GridReader.cube` assembles the
matched set into one store using the best method for that kind — radar stacks
timesteps into ``(time, y, x)`` incrementally, GRIB2 promotes its varying
forecast axes into an N-D cube via combine_by_coords. Both paths download to the
local bronze cache and read from disk (the tabular ``to_parquet()`` analog). A
cloud-lazy GRIB2 path (kerchunk/VirtualiZarr, to avoid loading the whole set
into memory) is future work.
"""

from __future__ import annotations

import contextlib
import logging
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from foehn.assets import Asset, assets_of, collection_assets, other_extensions
from foehn.collections import COLLECTION_META, COLLECTIONS
from foehn.fetch import Fetcher, FetchError
from foehn.transfer import exists, fetch_all
from foehn.workspace import Workspace

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import xarray as xr


# --- Optional dependencies -------------------------------------------------


def _require_xarray():
    """Import xarray, or raise a helpful error pointing at the optional extra."""
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError(
            "Reading gridded data requires the optional 'grids' dependencies "
            "(xarray + h5netcdf). Install them with:\n\n"
            '  pip install "foehn[grids]"\n'
        ) from exc
    return xr


def require_netcdf() -> None:
    """A :data:`Require`: NetCDF needs only xarray."""
    _require_xarray()


def require_grib2() -> None:
    """A :data:`Require`: GRIB2 needs xarray plus cfgrib + eccodes."""
    _require_xarray()
    try:
        import cfgrib  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Reading GRIB2 forecasts requires cfgrib + eccodes, part of the optional "
            "'grids' dependencies. Install them with:\n\n"
            '  pip install "foehn[grids]"\n'
        ) from exc


def require_radar() -> None:
    """A :data:`Require`: ODIM composites need xarray plus h5py + pyproj."""
    _require_xarray()
    try:
        import h5py  # noqa: F401
        import pyproj  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Reading HDF5/ODIM radar composites requires h5py + pyproj, part of the optional "
            "'grids' dependencies. Install them with:\n\n"
            '  pip install "foehn[grids]"\n'
        ) from exc


# --- The adapter -----------------------------------------------------------


Require = Callable[[], None]
"""Raise ImportError if this kind's optional dependencies are missing.

A separate field rather than the first line of :data:`OpenAdapter` on purpose:
the registry calls it *before* fetching anything, so a missing cfgrib fails in
milliseconds instead of after a 900 MB download.
"""


class OpenAdapter(Protocol):
    """How one grid kind opens a matched set of local files.

    Every adapter takes the same arguments and ignores what its kind does not
    use — the same convention as :class:`~foehn.registry.DownloadAdapter`, and
    for the same reason: the alternative is a per-kind call shape, which is the
    ladder this replaces.
    """

    def __call__(
        self,
        files: list[Path],
        *,
        dataset: str,
        workspace: Workspace,
        fetcher: Fetcher,
    ) -> xr.Dataset: ...


class CubeAdapter(Protocol):
    """How one grid kind assembles a matched set into a single Zarr store.

    Writes rather than returns, which is not symmetry for its own sake: the
    radar cube appends one timestep at a time so peak memory stays at a single
    file however many timesteps the match spans. An adapter that returned a
    Dataset would have to materialise the lot first.
    """

    def __call__(
        self,
        files: list[Path],
        store: Path,
        *,
        dataset: str,
        workspace: Workspace,
        fetcher: Fetcher,
        variables: str | list[str] | None,
        mode: str,
    ) -> None: ...


@dataclass(frozen=True)
class GridReader:
    """How one grid kind is read. Constructed in :data:`~foehn.registry.KINDS`.

    The per-kind functions live here beside the implementation they call; the
    configuration is stated in the registry row, exactly as ``key_columns`` and
    ``sort_column`` are for the tabular readers.
    """

    suffixes: tuple[str, ...]
    """Asset extensions this kind reads. Other payloads (GeoTIFF/ZIP copies) are ignored."""

    require: Require
    """Checked before anything is fetched."""

    open: OpenAdapter
    """Opens the matched files as one Dataset."""

    cube: CubeAdapter | None = None
    """Assembles the matched set into one store, or None for a kind that needs no
    cube builder — NetCDF combines a multi-file ``match`` on read already."""

    max_files: int | None = None
    """How many matched files one open may combine.

    None for NetCDF, which combines cleanly via combine_by_coords. GRIB2 and HDF5
    are capped at 1: ICON forecasts are on an unstructured grid combine_by_coords
    can't stack, and radar is one Cartesian composite per timestep (thousands per
    collection). The cap is checked against the *remote* listing before
    downloading, so an over-broad match is rejected even when one matching file
    already happens to be cached.
    """

    cube_max_files: int | None = None
    """The same cap for the cube path, which is a different number per kind.

    GRIB2 holds the whole set in memory at once, so it is capped; radar appends
    incrementally and deliberately wants every timestep the match spans.
    """

    match_example: str | None = None
    """Seeds the "narrow your match" guidance for the capped kinds."""

    cube_match_example: str | None = None
    """The same seed for the cube path, where a useful match is a wider one.

    A single radar file is ``cpc2613000000``; a radar cube wants the day prefix
    ``cpc26130``. Different advice, so a different field.
    """

    run_datetime: bool = False
    """Whether a STAC item's ``datetime`` is the model run embedded in the filename.

    True for GRIB2 only. The CSV and radar collections set ``datetime`` to a
    catalog-refresh timestamp unrelated to the data, where filtering on a real
    data date matches nothing.
    """


# --- Listing and fetching --------------------------------------------------


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
            f"which exceeds the {max_files}-file cap for cubing (the whole set is loaded into memory "
            "at once). Narrow match= to a smaller set"
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
    files missing from disk are actually downloaded. If the listing is
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

    # ``on_error="raise"`` rather than the download paths' count-and-continue: a
    # grid read cannot proceed on a partial set, so the first failure is fatal.
    # Destination de-duplication and the worker pool are the transfer module's.
    fetch_all(
        matched,
        out_dir,
        fetcher=fetcher,
        skip=exists,
        on_error="raise",
        label="grid file",
    )
    return sorted({out_dir / asset.name for asset in matched})


# --- Writing ---------------------------------------------------------------


# CF-compliant temporal base units that xarray/cftime can decode. Anything else
# in a "<unit> since <ref>" string (notably "years"/"months", used by MeteoSwiss
# climate normals) breaks CF decoding.
_CF_TIME_UNITS = frozenset(
    {"microseconds", "milliseconds", "seconds", "minutes", "hours", "days", "us", "ms", "s", "min", "hr", "h", "d"}
)


def sanitize_noncf_time_units(ds):
    """Rename non-CF temporal unit attrs so the dataset can be re-decoded later.

    A store written with a ``years since ...`` units attribute throws on every
    subsequent ``xr.open_zarr``. We move such units/calendar attrs aside (raw
    values are untouched) so the artifact opens cleanly.
    """
    for var in ds.variables.values():
        units = var.attrs.get("units")
        if isinstance(units, str) and " since " in units:
            head = units.split(" since ", 1)[0].strip().lower()
            if head not in _CF_TIME_UNITS:
                var.attrs["units_noncf"] = var.attrs.pop("units")
                if "calendar" in var.attrs:
                    var.attrs["calendar_noncf"] = var.attrs.pop("calendar")
                var.encoding.pop("units", None)
                var.encoding.pop("calendar", None)
    return ds


def write_zarr(ds, store: Path, mode: str, append_dim: str | None = None) -> None:
    """Write a Dataset to a Zarr store, suppressing only zarr's consolidated-metadata notice."""
    with warnings.catch_warnings():
        # xarray writes consolidated metadata by default. zarr-python 3 warns that
        # this is outside the Zarr v3 spec, but we keep it on purpose: it's purely
        # additive (every per-array zarr.json is still written), makes the common
        # open_zarr() path fast and warning-free, and zarr-python reads it back
        # natively. Suppress only that one deliberate warning — nothing else.
        warnings.filterwarnings(
            "ignore",
            message="Consolidated metadata is currently not part in the Zarr format 3 specification",
        )
        if append_dim is not None:
            ds.to_zarr(store, mode=mode, append_dim=append_dim)
        else:
            ds.to_zarr(store, mode=mode)


def select_variables(ds, variables: str | list[str] | None):
    """Restrict a Dataset to the requested data variable(s), or return it whole.

    Uniform across kinds, so the registry applies it once to whatever an open
    adapter returned rather than each adapter restating it.
    """
    if variables is None:
        return ds
    return ds[[variables] if isinstance(variables, str) else list(variables)]


# --- xarray-backed opens ---------------------------------------------------


def _open_grid(xr, files: list[Path], engine: str | None, backend_kwargs: dict | None = None):
    """Open one or many grid files (NetCDF or GRIB2) into a single Dataset.

    Multiple files are opened individually and merged with ``combine_by_coords``
    rather than ``open_mfdataset``: the latter needs a dask chunk manager, which
    is not in the 'grids' extra. Per-file opens keep the backend's own
    lazy reads, so combining stays dask-free.

    ``combine_attrs="drop_conflicts"`` keeps global attributes shared by every
    file (title, institution, …) but drops those that legitimately differ
    between files (history, source, creation date). The default ("no_conflicts")
    raises a MergeError on any such difference, even when the arrays combine
    cleanly — which is common across a real multi-file MeteoSwiss set.

    Retries with ``decode_times=False`` when the source uses non-CF time units
    that xarray/cftime cannot decode — MeteoSwiss climate normals, for example,
    label their time axis "years since 1991-01-01", which is not CF-compliant.
    """

    def _do(decode_times: bool):
        kwargs: dict = {"engine": engine, "decode_times": decode_times}
        if backend_kwargs:
            kwargs["backend_kwargs"] = backend_kwargs
        if len(files) == 1:
            return xr.open_dataset(files[0], **kwargs)
        datasets = [xr.open_dataset(f, **kwargs) for f in files]
        return xr.combine_by_coords(datasets, combine_attrs="drop_conflicts")

    try:
        return _do(decode_times=True)
    except (ValueError, OverflowError) as exc:
        # Non-CF time units (e.g. "years since 1991-01-01") surface as a ValueError
        # from the CF time decoder; an out-of-range reference can raise OverflowError.
        # Narrow to those so an unrelated read error isn't masked behind the retry,
        # and chain the original cause if the fallback also fails.
        logger.debug("CF time decoding failed (%s); retrying with decode_times=False", exc)
        try:
            return _do(decode_times=False)
        except Exception as retry_exc:
            raise retry_exc from exc


# --- ODIM radar ------------------------------------------------------------


def _attr(group, key, default=None):
    """Read an HDF5 attribute, decoding bytes to str."""
    val = group.attrs.get(key, default)
    return val.decode() if isinstance(val, bytes) else val


def _open_odim_composite(xr, path: Path):
    """Read a MeteoSwiss ODIM-H5 Cartesian radar composite into an xarray Dataset.

    The OGD radar products (CombiPrecip precipitation, hail) are ODIM ``COMP``
    images, not polar volumes — a single 2-D grid at ``/dataset1/data1/data`` on
    the Swiss projection. We apply the ODIM ``gain``/``offset`` scaling, map the
    ``nodata`` sentinel (outside radar coverage) to NaN and ``undetect`` (nothing
    detected) to 0, and turn the ``/where`` projection metadata into LV95 x/y
    coordinates via pyproj so the result lines up with the NetCDF Swiss grids.
    """
    import h5py
    import numpy as np
    import pyproj

    with h5py.File(path, "r") as f:
        obj = _attr(f["what"], "object", "")
        if obj != "COMP":
            raise ValueError(f"{path.name}: expected an ODIM 'COMP' composite, got object={obj!r}.")

        node = f["dataset1"]["data1"]
        dwhat = node["what"]
        gain = float(_attr(dwhat, "gain", 1.0))
        offset = float(_attr(dwhat, "offset", 0.0))
        nodata = float(_attr(dwhat, "nodata", np.nan))
        undetect = float(_attr(dwhat, "undetect", np.nan))
        quantity = str(_attr(dwhat, "quantity", "data"))
        raw = node["data"][:].astype("float64")

        where = f["where"]
        xsize, ysize = int(where.attrs["xsize"]), int(where.attrs["ysize"])
        xscale, yscale = float(where.attrs["xscale"]), float(where.attrs["yscale"])
        projdef = _attr(where, "projdef")
        ul_lon, ul_lat = float(where.attrs["UL_lon"]), float(where.attrs["UL_lat"])

        what = f["what"]
        date, time = _attr(what, "date", ""), _attr(what, "time", "")
        long_name = ""
        if "how" in f and "MeteoSwiss" in f["how"]:
            long_name = _attr(f["how"]["MeteoSwiss"], "long_name", "")

    # Physical values: scale, then mask the ODIM sentinels (NaN/Inf-safe).
    values = offset + gain * raw
    nodata_mask = np.isnan(raw) if np.isnan(nodata) else (raw == nodata)
    values[nodata_mask] = np.nan
    if np.isinf(undetect):
        values[np.isinf(raw)] = 0.0
    elif not np.isnan(undetect):
        values[raw == undetect] = 0.0

    # Swiss LV95 cell-centre coordinates. The grid is axis-aligned in the ODIM
    # projection, so transforming the upper-left corner gives the origin; row 0
    # is the northernmost row (y decreases with row index).
    transformer = pyproj.Transformer.from_crs("EPSG:4326", pyproj.CRS.from_proj4(projdef), always_xy=True)
    x0, y0 = transformer.transform(ul_lon, ul_lat)
    x = x0 + (np.arange(xsize) + 0.5) * xscale
    y = y0 - (np.arange(ysize) + 0.5) * yscale

    var_attrs = {"quantity": quantity}
    if long_name:
        var_attrs["long_name"] = long_name
    da = xr.DataArray(values, dims=("y", "x"), coords={"y": y, "x": x}, name=quantity.lower(), attrs=var_attrs)
    ds = da.to_dataset()
    ds.coords["x"].attrs.update({"units": "m", "long_name": "Swiss LV95 easting (CHX)"})
    ds.coords["y"].attrs.update({"units": "m", "long_name": "Swiss LV95 northing (CHY)"})
    ds.attrs.update({"projdef": projdef, "grid": "swiss_lv95", "odim_object": obj})
    if date and time:
        with contextlib.suppress(ValueError, IndexError):
            ts = np.datetime64(f"{date[:4]}-{date[4:6]}-{date[6:8]}T{time[:2]}:{time[2:4]}:{time[4:6]}")
            ds = ds.assign_coords(time=ts)
    return ds


# --- ICON unstructured coordinates -----------------------------------------


# Parsed ICON/KENDA cell lat/lon — the constants GRIB is ~11 MB and the same grid
# for every field in a collection, so parse it once. Keyed by (dataset,
# workspace root): the constants file is resolved inside the workspace, so
# keying on the dataset alone hands a second workspace the first one's
# coordinates.
_ICON_COORDS_CACHE: dict[tuple[str, str], tuple] = {}


def _ensure_constants_file(dataset: str, workspace: Workspace, *, fetcher: Fetcher) -> Path | None:
    """Locate (or download) a GRIB2 collection's horizontal-constants file.

    Returns the local path, or None if the collection exposes no such asset.
    The constants file is a collection-level STAC asset (not a per-item one).
    """
    out_dir = workspace.bronze(dataset)
    cached = sorted(out_dir.glob("horizontal_constants*.grib2"))
    if cached:
        return cached[0]

    meta = fetcher.collection(COLLECTIONS[dataset])
    constants = collection_assets(meta, key_contains="horizontal_constants")
    if not constants:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    href = constants[0].href
    path = out_dir / constants[0].name
    if not path.exists():
        fetcher.stream(href, path)
    return path


def _icon_unstructured_lonlat(dataset: str, workspace: Workspace, *, fetcher: Fetcher):
    """Return (lat, lon) cell-centre arrays for an ICON/KENDA grid, or (None, None).

    Read from the collection's horizontal-constants GRIB (``tlat``/``tlon`` on
    the same ``values`` dimension as the forecast fields). Cached per collection.
    """
    cache_key = (dataset, str(workspace.root))
    if cache_key in _ICON_COORDS_CACHE:
        return _ICON_COORDS_CACHE[cache_key]

    import cfgrib

    lat = lon = None
    path = _ensure_constants_file(dataset, workspace, fetcher=fetcher)
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
        _ICON_COORDS_CACHE[cache_key] = (lat, lon)
    return lat, lon


def _attach_icon_lonlat(ds, dataset: str, workspace: Workspace, *, fetcher: Fetcher, what: str):
    """Attach ICON cell lat/lon to an unstructured Dataset, best-effort.

    ICON/KENDA are on an unstructured grid (1-D ``values``, no lat/lon in the
    GRIB), so the coordinates come from the collection's constants file. Warn and
    continue if it is unavailable (e.g. offline) rather than failing an
    otherwise-good read.
    """
    if "values" not in ds.dims or "lat" in ds.coords:
        return ds
    try:
        lat, lon = _icon_unstructured_lonlat(dataset, workspace, fetcher=fetcher)
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


# --- The open adapters -----------------------------------------------------


def open_netcdf(files: list[Path], *, dataset: str, **_: object) -> xr.Dataset:
    """Open one or more NetCDF files, combining them on their coordinates."""
    xr = _require_xarray()
    try:
        return _open_grid(xr, files, engine=None)
    except Exception as exc:
        if len(files) > 1:
            fmt = COLLECTION_META[dataset]["format"]
            raise ValueError(
                f"Could not combine the {len(files)} {fmt} files for {dataset!r} into one Dataset "
                f"({type(exc).__name__}: {exc}). This set mixes parameters/levels/resolutions — "
                "narrow to a coherent set with match=, e.g. "
                f'foehn.open_dataset({dataset!r}, match="<parameter>").'
            ) from exc
        raise


def open_grib2(files: list[Path], *, dataset: str, workspace: Workspace, fetcher: Fetcher, **_: object) -> xr.Dataset:
    """Open one GRIB2 field via cfgrib, geo-referenced onto the ICON cell grid."""
    xr = _require_xarray()
    ds = _open_grid(xr, files, engine="cfgrib", backend_kwargs={"indexpath": ""})
    return _attach_icon_lonlat(
        ds,
        dataset,
        workspace,
        fetcher=fetcher,
        what="returning the unstructured grid without coordinates.",
    )


def open_radar(files: list[Path], **_: object) -> xr.Dataset:
    """Open one ODIM-H5 Cartesian composite."""
    xr = _require_xarray()
    return _open_odim_composite(xr, files[0])


# --- The cube adapters -----------------------------------------------------


# Fixed time encoding for stacked writes. Incremental Zarr appends must share one
# epoch — otherwise xarray re-infers per-frame units on each append and the time
# axis is silently corrupted (e.g. 5-minute steps decoded as 5-day steps).
_STACK_TIME_ENCODING = {"units": "seconds since 1970-01-01", "calendar": "proleptic_gregorian", "dtype": "int64"}


def cube_radar(
    files: list[Path],
    store: Path,
    *,
    variables: str | list[str] | None = None,
    mode: str = "w",
    **_: object,
) -> None:
    """Stack matched radar composites into one ``(time, y, x)`` Zarr cube.

    Written incrementally — one timestep appended at a time along ``time`` — so
    peak memory stays at a single file no matter how many timesteps the match
    spans, and no dask is needed.
    """
    xr = _require_xarray()

    wanted = None
    if variables is not None:
        wanted = [variables] if isinstance(variables, str) else list(variables)

    # Radar filenames embed a zero-padded timestamp, so lexical order is chronological.
    for i, path in enumerate(sorted(files)):
        ds = _open_odim_composite(xr, path)
        if "time" not in ds.coords:
            raise ValueError(f"{path.name}: no time coordinate — cannot stack along time.")
        if wanted is not None:
            ds = ds[wanted]
        ds = sanitize_noncf_time_units(ds).expand_dims("time")
        ds["time"].encoding.update(_STACK_TIME_ENCODING)
        write_zarr(ds, store, mode if i == 0 else "a", append_dim=None if i == 0 else "time")


def cube_grib2(
    files: list[Path],
    store: Path,
    *,
    dataset: str,
    workspace: Workspace,
    fetcher: Fetcher,
    variables: str | list[str] | None = None,
    mode: str = "w",
) -> None:
    """Combine matched GRIB2 files into one N-D cube over their varying axes.

    Each ICON/KENDA file is a single (variable, member, lead time, reference time)
    field on the unstructured ``values`` grid. Whichever of ``number``/``time``/
    ``step`` differ across the matched set are promoted to dimensions and merged
    with ``combine_by_coords`` into a cube (e.g. ``(time, step, values)``). The
    derived ``valid_time`` is dropped before combining (it conflicts on concat)
    and recomputed afterwards. The whole set is loaded into memory at once.
    """
    xr = _require_xarray()

    datasets = [xr.open_dataset(f, engine="cfgrib", backend_kwargs={"indexpath": ""}) for f in sorted(files)]
    # Promote whichever independent axes actually differ across the matched files.
    varying = [
        coord
        for coord in ("number", "time", "step")
        if len({ds[coord].values.tobytes() for ds in datasets if coord in ds.coords}) > 1
    ]
    if not varying:
        raise ValueError(
            f"The match selected files that don't differ in number/time/step for {dataset!r} — "
            "nothing to assemble into a cube (open it as a single field instead)."
        )
    prepared = [ds.drop_vars("valid_time", errors="ignore").expand_dims(varying) for ds in datasets]
    cube = xr.combine_by_coords(prepared, combine_attrs="drop_conflicts")
    if "time" in cube.coords and "step" in cube.coords:
        cube = cube.assign_coords(valid_time=cube["time"] + cube["step"])

    cube = select_variables(cube, variables)
    cube = _attach_icon_lonlat(
        cube,
        dataset,
        workspace,
        fetcher=fetcher,
        what="the cube is on the bare unstructured grid.",
    )
    write_zarr(sanitize_noncf_time_units(cube), store, mode)


__all__ = [
    "CubeAdapter",
    "GridReader",
    "OpenAdapter",
    "Require",
    "cube_grib2",
    "cube_radar",
    "ensure_grid_files",
    "open_grib2",
    "open_netcdf",
    "open_radar",
    "require_grib2",
    "require_netcdf",
    "require_radar",
    "sanitize_noncf_time_units",
    "select_variables",
    "write_zarr",
]
