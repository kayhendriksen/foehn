"""Open MeteoSwiss gridded collections as xarray Datasets.

This is the gridded counterpart to the tabular ``foehn.load()`` path: where
``load()`` returns a Polars DataFrame for CSV station data, ``open_dataset()``
returns an xarray Dataset for the N-dimensional grid collections.

PHASE 1 — NetCDF only
---------------------
Only the NetCDF collections (``NETCDF_COLLECTIONS``) are wired up. xarray reads
NetCDF4/HDF5 natively, so no Zarr conversion is involved — files are downloaded
to the local bronze cache once and opened lazily from disk.

GRIB2 (ICON/KENDA) and HDF5 radar (ODIM) are deliberately not handled yet: GRIB2
needs cfgrib/eccodes and benefits from a kerchunk/VirtualiZarr consolidation step,
and ODIM radar needs ``xradar`` to map cleanly onto xarray. Those land in later
phases. ``to_zarr()`` (writing a user-visible Zarr store under
``data_dir/zarr/<key>/``) is the Phase 2 follow-up.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from foehn._urls import validate_download_href
from foehn.collections import COLLECTIONS, GRIB2_COLLECTIONS, NETCDF_COLLECTIONS
from foehn.stac import get_collection_items

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


def _ensure_netcdf_files(collection_key: str, bronze_dir: Path, match: str | None = None) -> list[Path]:
    """Return local .nc files for a collection, downloading them if absent.

    Only NetCDF assets are fetched — GeoTIFF/ZIP payloads (which for these
    collections are redundant copies of the .nc data) are ignored. ``match``
    keeps only files whose name contains the given substring, which is how
    callers narrow a heterogeneous multi-file collection to a coherent set.
    """
    out_dir = bronze_dir / collection_key
    local = sorted(out_dir.glob("*.nc"))
    if local:
        files = [f for f in local if match is None or match in f.name]
        if files:
            return files
        # Local cache exists but nothing matches — fall through to the network
        # path in case the requested files were simply never downloaded.

    # Lazy import to avoid a hard dependency on requests at module import time
    # for callers that only touch the registry helpers.
    from foehn.client import _download_binary, _retry_session

    collection_id = COLLECTIONS[collection_key]
    items = get_collection_items(collection_id, require_csv=False, verbose=False)

    nc_hrefs: list[str] = []
    other_exts: set[str] = set()
    for item in items:
        for asset_info in item.get("assets", {}).values():
            href = asset_info.get("href", "")
            clean = href.split("?")[0]
            filename = clean.split("/")[-1]
            if clean.endswith(".nc"):
                if match is None or match in filename:
                    nc_hrefs.append(href)
            elif "." in filename:
                other_exts.add("." + filename.rsplit(".", 1)[-1])

    if not nc_hrefs:
        if match is not None:
            raise ValueError(f"No NetCDF (.nc) assets matching {match!r} found for {collection_key!r}.")
        found = ", ".join(sorted(other_exts)) or "none"
        raise ValueError(
            f"No NetCDF (.nc) assets found for {collection_key!r} "
            f"(available asset types: {found}). open_dataset() supports NetCDF only; "
            "GeoTIFF/ZIP/GRIB2 are not handled."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    session = _retry_session()
    paths: list[Path] = []
    for href in nc_hrefs:
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


def _open_netcdf(xr, files: list[Path], engine: str | None):
    """Open one or many NetCDF files into a single Dataset.

    Retries with ``decode_times=False`` when the source uses non-CF time units
    that xarray/cftime cannot decode — MeteoSwiss climate normals, for example,
    label their time axis "years since 1991-01-01", which is not CF-compliant.
    """
    multi = len(files) > 1

    def _do(decode_times: bool):
        if multi:
            return xr.open_mfdataset(files, engine=engine, combine="by_coords", decode_times=decode_times)
        return xr.open_dataset(files[0], engine=engine, decode_times=decode_times)

    try:
        return _do(decode_times=True)
    except Exception:
        return _do(decode_times=False)


def _validate_grid_dataset(dataset: str) -> None:
    """Guard shared by open_dataset/to_zarr: only NetCDF collections are allowed."""
    if dataset not in COLLECTIONS:
        raise ValueError(f"Unknown dataset: {dataset!r}. Use list_datasets() to see available datasets.")
    if dataset not in NETCDF_COLLECTIONS:
        if dataset in GRIB2_COLLECTIONS:
            raise NotImplementedError(
                f"Dataset {dataset!r} is a GRIB2/HDF5 grid. open_dataset() supports NetCDF only "
                f"in this release. Download the raw files with the CLI: foehn download {dataset} --grids"
            )
        raise ValueError(f"Dataset {dataset!r} is tabular (CSV). Use foehn.load() to get a Polars DataFrame instead.")


def open_dataset(
    dataset: str,
    *,
    variables: str | list[str] | None = None,
    match: str | None = None,
    data_dir: Path | str | None = None,
    engine: str | None = None,
) -> xr.Dataset:
    """Open a gridded dataset as an xarray Dataset.

    The grid analog of ``foehn.load()``. NetCDF files are cached under
    ``data_dir/bronze/<dataset>/`` on first use and opened lazily from disk;
    subsequent calls reuse the cache.

    Args:
        dataset: Dataset name (e.g. "surface_derived_grid"). Use list_datasets()
            to see options. Must be a NetCDF collection.
        variables: Restrict to these data variable(s). If None, all are kept.
        match: Keep only source files whose name contains this substring. Use
            this to narrow a heterogeneous multi-file collection (e.g. different
            parameters or grid resolutions) to one coherent set that can be
            combined — analogous to the station/frequency filters on load().
        data_dir: Root data directory. Defaults to ./data/meteoswiss.
        engine: xarray backend engine. Default None lets xarray auto-detect —
            MeteoSwiss serves a mix of NetCDF-3 classic and NetCDF-4/HDF5, and
            netCDF4 (shipped in the 'grids' extra) reads both.

    Returns:
        An xarray Dataset. Reading is lazy — data is pulled from disk on demand.

    Raises:
        ValueError: If the dataset is unknown, tabular (CSV), or if its files
            cannot be combined into a single Dataset (narrow it with ``match``).
        NotImplementedError: If the dataset is a GRIB2/HDF5 grid (later phase).
        ImportError: If the optional 'grids' dependencies are not installed.

    Example::

        import foehn

        # A coherent single-parameter slice of a multi-file collection
        ds = foehn.open_dataset("surface_derived_grid", match="rhiresd")
        ds = foehn.open_dataset("climate_scenarios_grid", match="_pr_", variables="pr")
    """
    _validate_grid_dataset(dataset)
    xr = _require_xarray()

    data_dir = Path(data_dir) if data_dir else Path.cwd() / "data" / "meteoswiss"
    bronze_dir = data_dir / "bronze"
    files = _ensure_netcdf_files(dataset, bronze_dir, match=match)

    try:
        ds = _open_netcdf(xr, files, engine)
    except Exception as exc:
        if len(files) > 1:
            raise ValueError(
                f"Could not combine the {len(files)} NetCDF files for {dataset!r} into one Dataset "
                f"({type(exc).__name__}: {exc}). This collection mixes parameters/resolutions — "
                "narrow to a coherent set with match=, e.g. "
                f'foehn.open_dataset({dataset!r}, match="<parameter>").'
            ) from exc
        raise

    if variables is not None:
        wanted = [variables] if isinstance(variables, str) else list(variables)
        ds = ds[wanted]

    return ds


def to_zarr(
    dataset: str,
    *,
    variables: str | list[str] | None = None,
    match: str | None = None,
    data_dir: Path | str | None = None,
    rechunk: dict[str, int] | None = None,
    mode: str = "w",
) -> Path:
    """Materialise a gridded dataset to a Zarr store on disk.

    The grid analog of ``foehn.to_parquet()``: reads the source NetCDF via
    ``open_dataset()`` and writes a single Zarr store to
    ``data_dir/zarr/<dataset>.zarr``.

    Args:
        dataset: Dataset name. Must be a NetCDF collection.
        variables: Restrict to these data variable(s) before writing.
        match: Narrow a multi-file collection to a coherent set (see open_dataset).
        data_dir: Root data directory. Defaults to ./data/meteoswiss.
        rechunk: Optional dim→chunk-size mapping applied before writing, e.g.
            ``{"time": 24}``. Requires ``dask`` to be installed.
        mode: Zarr write mode (default "w" — overwrite any existing store).

    Returns:
        Path to the written ``.zarr`` store.
    """
    ds = open_dataset(dataset, variables=variables, match=match, data_dir=data_dir)
    ds = _sanitize_noncf_time_units(ds)

    data_dir = Path(data_dir) if data_dir else Path.cwd() / "data" / "meteoswiss"
    store = data_dir / "zarr" / f"{dataset}.zarr"
    store.parent.mkdir(parents=True, exist_ok=True)

    if rechunk:
        ds = ds.chunk(rechunk)

    ds.to_zarr(store, mode=mode)
    return store
