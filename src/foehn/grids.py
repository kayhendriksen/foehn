"""Open MeteoSwiss gridded collections as xarray Datasets.

This is the gridded counterpart to the tabular ``foehn.load()`` path: where
``load()`` returns a Polars DataFrame for CSV station data, ``open_dataset()``
returns an xarray Dataset for the N-dimensional grid collections.

Supported formats
-----------------
Routing keys off each collection's ``format`` (see ``_GRID_READERS``):

* NetCDF (climate grids, normals, scenarios) — xarray auto-detects the engine.
* GRIB2 (ICON-CH1/CH2 forecasts, KENDA analysis) — cfgrib engine.
* HDF5/ODIM radar composites (CombiPrecip, hail) — bespoke ODIM reader. These are
  Cartesian ``COMP`` grids (not polar volumes, so not xradar): read via h5py with
  ODIM gain/offset scaling onto Swiss LV95 x/y coordinates.

All formats install via the single optional 'grids' extra (``pip install
"foehn[grids]"``). GRIB2 and radar collections hold thousands of single-field
files, so ``open_dataset`` requires a ``match`` that resolves to one file (the
cap is enforced before downloading). ICON GRIB2 comes back on a 1-D ``values``
dimension; cell ``lat``/``lon`` are joined from the collection's horizontal-
constants file (best-effort), so the unstructured grid is geo-referenced.

``open_dataset()`` reads a single file; ``to_zarr(..., stack="auto")`` assembles
the matched set into one cube using the best method per format — radar stacks
timesteps into ``(time, y, x)`` incrementally, GRIB2 promotes its varying
forecast axes into an N-D cube via combine_by_coords. ``open_dataset()`` and
``to_zarr()`` download to the local bronze cache and read from disk (the tabular
``to_parquet()`` analog). A cloud-lazy GRIB2 path (kerchunk/VirtualiZarr, to
avoid loading the whole set into memory) is future work.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from foehn._urls import validate_download_href
from foehn.collections import COLLECTION_META, COLLECTIONS
from foehn.stac import get_collection_items, get_collection_metadata

if TYPE_CHECKING:
    import xarray as xr


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


def _require_cfgrib():
    """Import cfgrib, or raise a helpful error pointing at the 'grids' extra."""
    try:
        import cfgrib  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Reading GRIB2 forecasts requires cfgrib + eccodes, part of the optional "
            "'grids' dependencies. Install them with:\n\n"
            '  pip install "foehn[grids]"\n'
        ) from exc


def _require_radar_deps():
    """Import h5py + pyproj, or raise a helpful error pointing at the 'grids' extra."""
    try:
        import h5py  # noqa: F401
        import pyproj  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Reading HDF5/ODIM radar composites requires h5py + pyproj, part of the optional "
            "'grids' dependencies. Install them with:\n\n"
            '  pip install "foehn[grids]"\n'
        ) from exc


# Per-format read configuration, keyed off COLLECTION_META's "format". Routing
# uses this rather than the GRIB2_COLLECTIONS set, which lumps GRIB2 and HDF5
# together. NetCDF keeps engine=None so xarray auto-detects NetCDF-3 vs -4/HDF5;
# GRIB2 forces cfgrib and disables its .idx sidecar files (indexpath=""); HDF5
# radar uses a bespoke ODIM-composite reader (``reader="odim"``) instead of an
# xarray engine.
#
# ``max_files`` caps how many matched files a single open may combine. NetCDF
# collections combine cleanly via combine_by_coords, so there's no cap. GRIB2 and
# HDF5 are capped at 1: ICON forecasts are on an unstructured grid that
# combine_by_coords can't stack, and radar is one Cartesian composite per
# timestep (thousands per collection). Stacking either along time is a later
# phase. ``match_example`` seeds the "narrow your match" guidance. The cap is
# enforced *before* downloading so an over-broad match can't pull a whole run.
_GRID_READERS: dict[str, dict] = {
    "NetCDF": {"suffixes": (".nc",), "engine": None, "backend_kwargs": None, "max_files": None, "match_example": None},
    "GRIB2": {
        "suffixes": (".grib2", ".grib"),
        "engine": "cfgrib",
        "backend_kwargs": {"indexpath": ""},
        "max_files": 1,
        "match_example": "202605231500-0-t_2m-ctrl",
    },
    "HDF5": {
        "suffixes": (".h5",),
        "reader": "odim",
        "engine": None,
        "backend_kwargs": None,
        "max_files": 1,
        "match_example": "cpc2613000000",
    },
}


def _raise_if_too_many(collection_key: str, match: str | None, names: list[str], max_files: int | None) -> None:
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
        f"match={match!r} matched {len(names)} files for {collection_key!r}, {detail}. "
        f"Matches include:\n{examples}{more}"
    )


def _ensure_grid_files(
    collection_key: str,
    bronze_dir: Path,
    suffixes: tuple[str, ...] = (".nc",),
    match: str | None = None,
    max_files: int | None = None,
) -> list[Path]:
    """Return local grid files for a collection, downloading them if absent.

    Only assets whose name ends in one of ``suffixes`` are fetched (``.nc`` for
    NetCDF, ``.grib2``/``.grib`` for GRIB2); other payloads (GeoTIFF/ZIP copies)
    are ignored. ``match`` keeps only files whose name contains the given
    substring, which is how callers narrow a heterogeneous multi-file collection
    to a coherent set.

    A filtered request for a single-file format (``max_files == 1``, i.e. GRIB2)
    is served straight from the local cache when a matching file exists — one
    file is a complete answer. Every other request — unfiltered, or a multi-file
    NetCDF match whose cache could be an incomplete leftover from an interrupted
    download — consults the remote listing first, so it never hands back a
    partial cache; only files missing from disk are actually downloaded. If the
    listing is unreachable, it falls back to the cache (with a warning) rather
    than failing a call that could be served locally.
    """
    out_dir = bronze_dir / collection_key
    local = sorted(f for s in suffixes for f in out_dir.glob(f"*{s}"))
    if match is not None and local and max_files == 1:
        # Single-file formats only: a matching cached file is the whole answer.
        # Multi-file formats fall through to the listing so a partial cache
        # (e.g. 2 of 5 files from an interrupted download) can't pass as complete.
        files = [f for f in local if match in f.name]
        if files:
            _raise_if_too_many(collection_key, match, [f.name for f in files], max_files)
            return files
        # Local cache exists but nothing matches — fall through to the network
        # path in case the requested files were simply never downloaded.

    # Lazy import to avoid a hard dependency on requests at module import time
    # for callers that only touch the registry helpers.
    import requests

    from foehn.client import _download_binary, _retry_session

    sfx = "/".join(suffixes)
    collection_id = COLLECTIONS[collection_key]
    try:
        items = get_collection_items(collection_id, require_csv=False, verbose=False)
    except requests.exceptions.RequestException as exc:
        cached = [f for f in local if match is None or match in f.name]
        if cached:
            warnings.warn(
                f"Could not reach the STAC API to verify the {collection_key!r} cache "
                f"({type(exc).__name__}); using {len(cached)} locally cached file(s), "
                "which may be an incomplete subset of the collection.",
                stacklevel=2,
            )
            return cached
        raise

    hrefs: list[str] = []
    other_exts: set[str] = set()
    for item in items:
        for asset_info in item.get("assets", {}).values():
            href = asset_info.get("href", "")
            clean = href.split("?")[0]
            filename = clean.split("/")[-1]
            if clean.endswith(suffixes):
                if match is None or match in filename:
                    hrefs.append(href)
            elif "." in filename:
                other_exts.add("." + filename.rsplit(".", 1)[-1])

    if not hrefs:
        if match is not None:
            raise ValueError(f"No {sfx} assets matching {match!r} found for {collection_key!r}.")
        found = ", ".join(sorted(other_exts)) or "none"
        raise ValueError(f"No {sfx} assets found for {collection_key!r} (available asset types: {found}).")

    # Enforce the per-format file cap before downloading so an over-broad match
    # (e.g. a whole forecast run) can't pull hundreds of files off the network.
    _raise_if_too_many(collection_key, match, [h.split("?")[0].split("/")[-1] for h in hrefs], max_files)

    out_dir.mkdir(parents=True, exist_ok=True)
    session = _retry_session()
    paths: list[Path] = []
    for href in hrefs:
        validate_download_href(href)
        filepath = out_dir / href.split("?")[0].split("/")[-1]
        if not filepath.exists():
            _download_binary(session, href, filepath)
        paths.append(filepath)
    return sorted(paths)


# CF-compliant temporal base units that xarray/cftime can decode. Anything else
# in a "<unit> since <ref>" string (notably "years"/"months", used by MeteoSwiss
# climate normals) breaks CF decoding.
_CF_TIME_UNITS = frozenset(
    {"microseconds", "milliseconds", "seconds", "minutes", "hours", "days", "us", "ms", "s", "min", "hr", "h", "d"}
)


def _sanitize_noncf_time_units(ds):
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
    except Exception:
        return _do(decode_times=False)


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
        try:
            ts = np.datetime64(f"{date[:4]}-{date[4:6]}-{date[6:8]}T{time[:2]}:{time[2:4]}:{time[4:6]}")
            ds = ds.assign_coords(time=ts)
        except (ValueError, IndexError):
            pass
    return ds


# Parsed ICON/KENDA cell lat/lon, keyed by collection — the constants GRIB is
# ~11 MB and the same grid for every field in a collection, so parse it once.
_ICON_COORDS_CACHE: dict[str, tuple] = {}


def _ensure_constants_file(collection_key: str, bronze_dir: Path) -> Path | None:
    """Locate (or download) a GRIB2 collection's horizontal-constants file.

    Returns the local path, or None if the collection exposes no such asset.
    The constants file is a collection-level STAC asset (not a per-item one).
    """
    out_dir = bronze_dir / collection_key
    cached = sorted(out_dir.glob("horizontal_constants*.grib2"))
    if cached:
        return cached[0]

    meta = get_collection_metadata(COLLECTIONS[collection_key])
    href = next(
        (a.get("href", "") for k, a in meta.get("assets", {}).items() if "horizontal_constants" in k.lower()),
        "",
    )
    if not href:
        return None

    from foehn.client import _download_binary, _retry_session

    validate_download_href(href)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / href.split("?")[0].split("/")[-1]
    if not path.exists():
        _download_binary(_retry_session(), href, path)
    return path


def _icon_unstructured_lonlat(collection_key: str, bronze_dir: Path):
    """Return (lat, lon) cell-centre arrays for an ICON/KENDA grid, or (None, None).

    Read from the collection's horizontal-constants GRIB (``tlat``/``tlon`` on
    the same ``values`` dimension as the forecast fields). Cached per collection.
    """
    if collection_key in _ICON_COORDS_CACHE:
        return _ICON_COORDS_CACHE[collection_key]

    import cfgrib

    lat = lon = None
    path = _ensure_constants_file(collection_key, bronze_dir)
    if path is not None:
        for ds in cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""}):
            if "tlat" in ds.variables:
                lat = ds["tlat"].values
            if "tlon" in ds.variables:
                lon = ds["tlon"].values
    _ICON_COORDS_CACHE[collection_key] = (lat, lon)
    return lat, lon


def _validate_grid_dataset(dataset: str) -> None:
    """Guard shared by open_dataset/to_zarr. Allows NetCDF/GRIB2/HDF5 grids; rejects tabular."""
    if dataset not in COLLECTIONS:
        raise ValueError(f"Unknown dataset: {dataset!r}. Use list_datasets() to see available datasets.")
    fmt = COLLECTION_META[dataset]["format"]
    if fmt in _GRID_READERS:
        return
    raise ValueError(f"Dataset {dataset!r} is tabular ({fmt}). Use foehn.load() to get a Polars DataFrame instead.")


def open_dataset(
    dataset: str,
    *,
    variables: str | list[str] | None = None,
    match: str | None = None,
    data_dir: Path | str | None = None,
    engine: str | None = None,
) -> xr.Dataset:
    """Open a gridded dataset as an xarray Dataset.

    The grid analog of ``foehn.load()``, for NetCDF collections (climate grids,
    normals, scenarios), GRIB2 forecasts (ICON-CH1/CH2, KENDA), and HDF5/ODIM
    radar composites (CombiPrecip, hail). This is *download-then-lazy*: the source
    file(s) are fetched in full to ``data_dir/bronze/<dataset>/`` on first use,
    then opened and read from that local copy. It is not cloud-lazy — there is no
    byte-range/partial read of the remote file, so the first call pays the full
    file size up front. Subsequent calls reuse the cache.

    GRIB2 and radar (HDF5) collections **require** ``match``, and it must resolve
    to a *single* file:

    * GRIB2 forecast collections hold thousands of files (one per variable ×
      ensemble member × lead time × reference time), and ICON's native
      unstructured (icosahedral) grid — a 1-D ``values`` dimension with no
      dimension coordinate — can't be stacked by ``combine_by_coords``. Include
      the reference + lead time, e.g. ``match="202605231500-0-t_2m-ctrl"``. The
      one field comes back on the ``values`` grid with cell ``lat``/``lon``
      coordinates joined from the collection's horizontal-constants file.
    * Radar collections hold one Cartesian composite per timestep (every ~5 min).
      Match a single file, e.g. ``match="cpc2613000000"``. The composite is read
      with ODIM gain/offset scaling, ``nodata`` masked to NaN, on Swiss LV95
      ``x``/``y`` coordinates (matching the NetCDF grids).

    ``open_dataset`` reads one field; to assemble many matched files into a cube
    use ``to_zarr(..., stack="auto")`` instead.

    Args:
        dataset: Dataset name (e.g. "surface_derived_grid", "forecast_icon_ch1",
            "radar_precip"). Use list_datasets() to see options. Must be a NetCDF,
            GRIB2, or HDF5/radar collection.
        variables: Restrict to these data variable(s). If None, all are kept.
        match: Keep only source files whose name contains this substring. Narrows
            a heterogeneous multi-file collection to one coherent set — analogous
            to the station/frequency filters on load(). Required for GRIB2 and
            radar collections, where it must select a single file.
        data_dir: Root data directory. Defaults to ./data/meteoswiss.
        engine: xarray backend engine. Default None auto-selects per format —
            xarray auto-detects NetCDF-3 vs NetCDF-4/HDF5, GRIB2 uses cfgrib, and
            radar uses a bespoke ODIM-composite reader (ignores ``engine``).

    Returns:
        An xarray Dataset backed by the local file(s), downloaded in full first
        (see the download-then-lazy note above) — e.g. the first
        ``climate_scenarios_grid`` call fetches ~900 MB before you read a pixel.

    Raises:
        ValueError: If the dataset is unknown, tabular (CSV), a GRIB2/radar
            collection opened without a single-file ``match``, or if its files
            cannot be combined into a single Dataset (narrow it with ``match``).
        ImportError: If the optional 'grids' dependencies are not installed.

    Example::

        import foehn

        # NetCDF: a coherent single-parameter slice of a multi-file collection
        ds = foehn.open_dataset("surface_derived_grid", match="rhiresd")
        ds = foehn.open_dataset("climate_scenarios_grid", match="_pr_", variables="pr")

        # GRIB2: a single forecast field — variable + member + reference + lead time
        ds = foehn.open_dataset("forecast_icon_ch1", match="202605231500-0-t_2m-ctrl")

        # Radar: a single CombiPrecip composite (one 5-min timestep)
        ds = foehn.open_dataset("radar_precip", match="cpc2613000000")
    """
    _validate_grid_dataset(dataset)
    fmt = COLLECTION_META[dataset]["format"]
    reader = _GRID_READERS[fmt]

    xr = _require_xarray()
    if fmt == "GRIB2":
        _require_cfgrib()
    elif fmt == "HDF5":
        _require_radar_deps()

    # Single-file formats (GRIB2, radar) have thousands of files per collection,
    # so an unfiltered open would download the lot — require a narrowing match.
    if reader["max_files"] == 1 and match is None:
        raise ValueError(
            f"Dataset {dataset!r} is a {fmt} collection of many single-field files; opening it "
            "unfiltered would download them all. Narrow to one file with match=, e.g. "
            f'foehn.open_dataset({dataset!r}, match="{reader["match_example"]}").'
        )

    data_dir = Path(data_dir) if data_dir else Path.cwd() / "data" / "meteoswiss"
    bronze_dir = data_dir / "bronze"
    files = _ensure_grid_files(
        dataset, bronze_dir, suffixes=reader["suffixes"], match=match, max_files=reader["max_files"]
    )

    if reader.get("reader") == "odim":
        ds = _open_odim_composite(xr, files[0])
    else:
        engine = engine if engine is not None else reader["engine"]
        try:
            ds = _open_grid(xr, files, engine, reader["backend_kwargs"])
        except Exception as exc:
            if len(files) > 1:
                raise ValueError(
                    f"Could not combine the {len(files)} {fmt} files for {dataset!r} into one Dataset "
                    f"({type(exc).__name__}: {exc}). This set mixes parameters/levels/resolutions — "
                    "narrow to a coherent set with match=, e.g. "
                    f'foehn.open_dataset({dataset!r}, match="<parameter>").'
                ) from exc
            raise

        # ICON/KENDA are on an unstructured grid (1-D ``values``, no lat/lon in
        # the GRIB). Attach cell lat/lon from the collection's constants file so
        # the data is geo-referenced. Best-effort: warn and continue if it's
        # unavailable (e.g. offline) rather than failing an otherwise-good read.
        if fmt == "GRIB2" and "values" in ds.dims and "lat" not in ds.coords:
            try:
                lat, lon = _icon_unstructured_lonlat(dataset, bronze_dir)
                if lat is not None and lon is not None and lat.size == ds.sizes["values"]:
                    ds = ds.assign_coords(
                        lat=("values", lat, {"units": "degrees_north", "standard_name": "latitude"}),
                        lon=("values", lon, {"units": "degrees_east", "standard_name": "longitude"}),
                    )
            except Exception as exc:
                warnings.warn(
                    f"Could not attach ICON lat/lon for {dataset!r} ({type(exc).__name__}); "
                    "returning the unstructured grid without coordinates.",
                    stacklevel=2,
                )

    if variables is not None:
        wanted = [variables] if isinstance(variables, str) else list(variables)
        ds = ds[wanted]

    return ds


def _store_slug(match: str) -> str:
    """Filesystem-safe fragment derived from a ``match`` filter for store names."""
    return re.sub(r"[^0-9A-Za-z]+", "_", match).strip("_") or "match"


def _resolve_store(dataset: str, match: str | None, data_dir, store) -> Path:
    """Resolve the .zarr output path: explicit ``store`` wins, else data_dir/zarr/<name>."""
    if store is not None:
        return Path(store)
    root = Path(data_dir) if data_dir else Path.cwd() / "data" / "meteoswiss"
    name = dataset if match is None else f"{dataset}__{_store_slug(match)}"
    return root / "zarr" / f"{name}.zarr"


# Fixed time encoding for stacked writes. Incremental Zarr appends must share one
# epoch — otherwise xarray re-infers per-frame units on each append and the time
# axis is silently corrupted (e.g. 5-minute steps decoded as 5-day steps).
_STACK_TIME_ENCODING = {"units": "seconds since 1970-01-01", "calendar": "proleptic_gregorian", "dtype": "int64"}


def _write_zarr(ds, store: Path, mode: str, append_dim: str | None = None) -> None:
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


def _to_zarr_stacked(dataset, fmt, *, variables, match, data_dir, store, mode) -> Path:
    """Stack a matched set of radar composites into one ``(time, y, x)`` Zarr cube.

    Written incrementally — one timestep appended at a time along ``time`` — so
    peak memory stays at a single file no matter how many timesteps the match
    spans, and no dask is needed.
    """
    if fmt != "HDF5":
        raise ValueError(
            f"stack='time' is only supported for radar (HDF5) collections; {dataset!r} is {fmt}. "
            "NetCDF multi-file matches already combine via match=; GRIB2 uses stack='auto'."
        )
    if match is None:
        raise ValueError(
            f"Stacking radar timesteps needs match= to scope the time range for {dataset!r} "
            '(e.g. a day prefix like match="cpc26130").'
        )

    xr = _require_xarray()
    _require_radar_deps()
    reader = _GRID_READERS[fmt]

    root = Path(data_dir) if data_dir else Path.cwd() / "data" / "meteoswiss"
    # No single-file cap here: stacking deliberately wants the whole matched set.
    files = _ensure_grid_files(dataset, root / "bronze", suffixes=reader["suffixes"], match=match)
    store_path = _resolve_store(dataset, match, data_dir, store)
    store_path.parent.mkdir(parents=True, exist_ok=True)

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
        ds = _sanitize_noncf_time_units(ds).expand_dims("time")
        ds["time"].encoding.update(_STACK_TIME_ENCODING)
        _write_zarr(ds, store_path, mode if i == 0 else "a", append_dim=None if i == 0 else "time")
    return store_path


# Cap on how many GRIB2 files a single stack="auto" cube may load. The whole set
# is held in memory at once (no dask), so this guards against an over-broad match
# OOM-ing the process; it's enforced before downloading.
_HYPERCUBE_MAX_FILES = 1000


def _to_zarr_hypercube(dataset, fmt, *, variables, match, data_dir, store, mode) -> Path:
    """Combine matched GRIB2 files into one N-D cube over their varying axes.

    Each ICON/KENDA file is a single (variable, member, lead time, reference time)
    field on the unstructured ``values`` grid. Whichever of ``number``/``time``/
    ``step`` differ across the matched set are promoted to dimensions and merged
    with ``combine_by_coords`` into a cube (e.g. ``(time, step, values)``). The
    derived ``valid_time`` is dropped before combining (it conflicts on concat)
    and recomputed afterwards. The whole set is loaded into memory at once.
    """
    if fmt != "GRIB2":
        raise ValueError(
            f"stack='auto' is only supported for GRIB2 collections; {dataset!r} is {fmt}. "
            "Radar uses stack='time'; NetCDF multi-file matches already combine via match=."
        )
    if match is None:
        raise ValueError(
            f"stack='auto' needs match= to scope the cube for {dataset!r}, e.g. "
            'match="-t_2m-ctrl" (one variable + member across runs and lead times).'
        )

    xr = _require_xarray()
    _require_cfgrib()
    reader = _GRID_READERS[fmt]

    root = Path(data_dir) if data_dir else Path.cwd() / "data" / "meteoswiss"
    bronze_dir = root / "bronze"
    files = _ensure_grid_files(
        dataset, bronze_dir, suffixes=reader["suffixes"], match=match, max_files=_HYPERCUBE_MAX_FILES
    )
    store_path = _resolve_store(dataset, match, data_dir, store)
    store_path.parent.mkdir(parents=True, exist_ok=True)

    datasets = [xr.open_dataset(f, engine="cfgrib", backend_kwargs=reader["backend_kwargs"]) for f in sorted(files)]
    # Promote whichever independent axes actually differ across the matched files.
    varying = [
        coord
        for coord in ("number", "time", "step")
        if len({ds[coord].values.tobytes() for ds in datasets if coord in ds.coords}) > 1
    ]
    if not varying:
        raise ValueError(
            f"match={match!r} selected files that don't differ in number/time/step — "
            "nothing to assemble into a cube (open it as a single field instead)."
        )
    prepared = [ds.drop_vars("valid_time", errors="ignore").expand_dims(varying) for ds in datasets]
    cube = xr.combine_by_coords(prepared, combine_attrs="drop_conflicts")
    if "time" in cube.coords and "step" in cube.coords:
        cube = cube.assign_coords(valid_time=cube["time"] + cube["step"])

    if variables is not None:
        cube = cube[[variables] if isinstance(variables, str) else list(variables)]

    if "values" in cube.dims and "lat" not in cube.coords:
        try:
            lat, lon = _icon_unstructured_lonlat(dataset, bronze_dir)
            if lat is not None and lon is not None and lat.size == cube.sizes["values"]:
                cube = cube.assign_coords(
                    lat=("values", lat, {"units": "degrees_north", "standard_name": "latitude"}),
                    lon=("values", lon, {"units": "degrees_east", "standard_name": "longitude"}),
                )
        except Exception as exc:
            warnings.warn(
                f"Could not attach ICON lat/lon for {dataset!r} ({type(exc).__name__}); "
                "the cube is on the bare unstructured grid.",
                stacklevel=2,
            )

    _write_zarr(_sanitize_noncf_time_units(cube), store_path, mode)
    return store_path


def to_zarr(
    dataset: str,
    *,
    variables: str | list[str] | None = None,
    match: str | None = None,
    data_dir: Path | str | None = None,
    store: Path | str | None = None,
    rechunk: dict[str, int] | None = None,
    mode: str = "w",
    stack: str | None = None,
) -> Path:
    """Materialise a gridded dataset to a Zarr store on disk.

    The grid analog of ``foehn.to_parquet()``: reads the source (NetCDF, GRIB2,
    or HDF5/radar) via ``open_dataset()`` and writes a single Zarr store under
    ``data_dir/zarr/``. (GRIB2 and radar collections require ``match`` — see
    ``open_dataset``.)

    The default store name encodes ``match`` so that different filtered slices of
    the same collection don't silently overwrite each other:
    ``<dataset>.zarr`` when unfiltered, ``<dataset>__<match>.zarr`` otherwise
    (e.g. ``surface_derived_grid__rhiresd.zarr``). Pass ``store`` for an explicit
    path that overrides this.

    Args:
        dataset: Dataset name. Must be a NetCDF, GRIB2, or HDF5/radar collection.
        variables: Restrict to these data variable(s) before writing.
        match: Narrow a multi-file collection to a coherent set (see open_dataset);
            required for GRIB2 and radar collections.
        data_dir: Root data directory. Defaults to ./data/meteoswiss.
        store: Explicit output path for the ``.zarr`` store. Overrides the
            derived ``data_dir/zarr/<name>.zarr`` location when given.
        rechunk: Optional dim→chunk-size mapping applied before writing, e.g.
            ``{"time": 24}``. Requires ``dask`` (not part of the 'grids' extra —
            install separately with ``pip install dask``); raises ImportError
            if it is missing. Not supported together with ``stack``.
        mode: Zarr write mode (default "w" — overwrite the store at this path).
            Note distinct ``match`` values map to distinct default paths, so this
            only overwrites a prior run of the *same* slice, not a different one.
        stack: ``"auto"`` assembles the matched files into one cube using the
            best method for the format — radar stacks CombiPrecip timesteps into
            a ``(time, y, x)`` cube incrementally (dask-free, one timestep in
            memory); GRIB2 promotes whichever of number/time/step vary into an
            N-D cube (e.g. ``(time, step, values)``) via ``combine_by_coords``
            (whole set in memory, capped at 1000 files); NetCDF needs nothing
            extra since a multi-file ``match`` already combines on read.
            ``"time"`` is the explicit radar path (same result as ``"auto"`` for
            radar). Default None reads a single file. Incompatible with ``rechunk``.

    Returns:
        Path to the written ``.zarr`` store.
    """
    _validate_grid_dataset(dataset)
    fmt = COLLECTION_META[dataset]["format"]

    if stack is not None:
        if stack not in ("auto", "time"):
            raise ValueError(f"stack={stack!r} is not supported; use 'auto' (any gridded format) or 'time' (radar).")
        if rechunk:
            raise ValueError("rechunk= is not supported with stack= (the cube is written separately).")
        kwargs = {"variables": variables, "match": match, "data_dir": data_dir, "store": store, "mode": mode}
        # "auto" routes to each format's best cube builder; "time" is the explicit radar path.
        if stack == "time" or fmt == "HDF5":
            return _to_zarr_stacked(dataset, fmt, **kwargs)  # radar: incremental (time, y, x)
        if fmt == "GRIB2":
            return _to_zarr_hypercube(dataset, fmt, **kwargs)  # forecasts: N-D combine_by_coords
        # NetCDF + stack="auto": open_dataset already combines a multi-file match,
        # so just fall through to the normal single-write path below.

    ds = open_dataset(dataset, variables=variables, match=match, data_dir=data_dir)
    ds = _sanitize_noncf_time_units(ds)

    store = _resolve_store(dataset, match, data_dir, store)
    store.parent.mkdir(parents=True, exist_ok=True)

    if rechunk:
        import importlib.util

        if importlib.util.find_spec("dask") is None:
            raise ImportError(
                "to_zarr(rechunk=...) requires dask, which is not part of the "
                "'grids' extra. Install it with:\n\n  pip install dask\n"
            )
        ds = ds.chunk(rechunk)

    _write_zarr(ds, store, mode)
    return store
