"""Open MeteoSwiss gridded collections as xarray Datasets.

This is the gridded counterpart to the tabular ``foehn.load()`` path: where
``load()`` returns a Polars DataFrame for CSV station data, ``open_dataset()``
returns an xarray Dataset for the N-dimensional grid collections.

Supported formats
-----------------
Routing keys off each collection's ``format`` (see ``_GRID_READERS``):

* NetCDF (climate grids, normals, scenarios) — engine auto-detected, 'grids' extra.
* GRIB2 (ICON-CH1/CH2 forecasts, KENDA analysis) — cfgrib engine, 'grib' extra.
  These collections hold thousands of single-field files, so ``open_dataset``
  requires a ``match`` that resolves to one file (the cap is enforced before
  downloading). ICON's native unstructured grid comes back as a 1-D ``values``
  dimension (no lat/lon joined); stacking lead/reference times into a time
  series (concat-along-step, kerchunk) is a later phase.

``open_dataset()`` downloads the source files to the local bronze cache once and
reads them lazily from disk; ``to_zarr()`` materialises a collection to a Zarr
store under ``data_dir/zarr/`` (the tabular ``to_parquet()`` analog).

HDF5 radar (ODIM) is not handled yet — it needs ``xradar`` to map cleanly onto
xarray, and lands in a later phase. A cloud-lazy GRIB2 path (kerchunk/
VirtualiZarr, to avoid the full per-run download) is also future work.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from foehn._urls import validate_download_href
from foehn.collections import COLLECTION_META, COLLECTIONS
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


def _require_cfgrib():
    """Import cfgrib, or raise a helpful error pointing at the 'grib' extra."""
    try:
        import cfgrib  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Reading GRIB2 forecasts requires the optional 'grib' dependencies "
            "(cfgrib + eccodes). Install them with:\n\n"
            '  pip install "foehn[grib]"\n'
        ) from exc


# Per-format read configuration, keyed off COLLECTION_META's "format". Routing
# uses this rather than the GRIB2_COLLECTIONS set, which also lumps in HDF5
# radar. NetCDF keeps engine=None so xarray auto-detects NetCDF-3 vs -4/HDF5;
# GRIB2 forces cfgrib and disables its .idx sidecar files (indexpath="").
#
# ``max_files`` caps how many matched files a single open may combine. NetCDF
# collections combine cleanly via combine_by_coords, so there's no cap. GRIB2 is
# capped at 1: ICON forecasts are on an unstructured grid (a 1-D ``values`` dim
# with no dimension coordinate and only scalar step/time coords), which
# combine_by_coords cannot stack — multi-file consolidation across lead times /
# reference times (concat-along-step, kerchunk) is a later phase. The cap is
# enforced *before* downloading so an over-broad match can't pull a whole run.
_GRID_READERS: dict[str, dict] = {
    "NetCDF": {"suffixes": (".nc",), "engine": None, "backend_kwargs": None, "max_files": None},
    "GRIB2": {
        "suffixes": (".grib2", ".grib"),
        "engine": "cfgrib",
        "backend_kwargs": {"indexpath": ""},
        "max_files": 1,
    },
}


def _raise_if_too_many(collection_key: str, match: str | None, names: list[str], max_files: int | None) -> None:
    """Refuse a match that resolves to more files than a single open can combine."""
    if max_files is None or len(names) <= max_files:
        return
    examples = "\n".join(f"  {n}" for n in sorted(names)[:4])
    more = "" if len(names) <= 4 else f"\n  ... and {len(names) - 4} more"
    raise ValueError(
        f"match={match!r} matched {len(names)} files for {collection_key!r}, but this collection "
        "is read one file at a time (multi-file consolidation across lead times / reference times "
        f"is a later phase). Narrow match= to pick a single file, e.g. one of:\n{examples}{more}"
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
    is not in the 'grids'/'grib' extras. Per-file opens keep the backend's own
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


def _validate_grid_dataset(dataset: str) -> None:
    """Guard shared by open_dataset/to_zarr. Allows NetCDF + GRIB2; rejects the rest."""
    if dataset not in COLLECTIONS:
        raise ValueError(f"Unknown dataset: {dataset!r}. Use list_datasets() to see available datasets.")
    fmt = COLLECTION_META[dataset]["format"]
    if fmt in _GRID_READERS:
        return
    if fmt == "HDF5":
        raise NotImplementedError(
            f"Dataset {dataset!r} is HDF5/ODIM radar. open_dataset() does not read radar yet "
            f"(needs xradar). Download the raw files with the CLI: foehn download {dataset} --grids"
        )
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
    normals, scenarios) and GRIB2 forecasts (ICON-CH1/CH2, KENDA). This is
    *download-then-lazy*: the source file(s) are fetched in full to
    ``data_dir/bronze/<dataset>/`` on first use, then opened and read lazily from
    that local copy. It is not cloud-lazy — there is no byte-range/partial read
    of the remote file, so the first call pays the full file size up front.
    Subsequent calls reuse the cache.

    GRIB2 collections **require** ``match``, and it must resolve to a *single*
    file. A forecast collection holds thousands of files (one per variable ×
    ensemble member × lead time × reference time), and ICON's native unstructured
    (icosahedral) grid — a 1-D ``values`` dimension with no dimension coordinate —
    can't be stacked by ``combine_by_coords``. So include the reference time and
    lead time, e.g. ``match="202605231500-0-t_2m-ctrl"``; cfgrib returns that one
    field with no lat/lon attached (the grid definition ships separately). Stacking
    lead times / reference times into a time series is a later phase.

    Args:
        dataset: Dataset name (e.g. "surface_derived_grid", "forecast_icon_ch1").
            Use list_datasets() to see options. Must be a NetCDF or GRIB2 collection.
        variables: Restrict to these data variable(s). If None, all are kept.
        match: Keep only source files whose name contains this substring. Narrows
            a heterogeneous multi-file collection to one coherent set that can be
            combined — analogous to the station/frequency filters on load().
            Required for GRIB2 collections.
        data_dir: Root data directory. Defaults to ./data/meteoswiss.
        engine: xarray backend engine. Default None auto-selects per format —
            xarray auto-detects NetCDF-3 vs NetCDF-4/HDF5, and GRIB2 uses cfgrib.

    Returns:
        An xarray Dataset backed by the local file(s). Array values are read
        lazily from the on-disk copy, but the file itself was already downloaded
        in full (see the download-then-lazy note above) — e.g. the first
        ``climate_scenarios_grid`` call fetches ~900 MB before you read a pixel.

    Raises:
        ValueError: If the dataset is unknown, tabular (CSV), a GRIB2 collection
            opened without ``match``, or if its files cannot be combined into a
            single Dataset (narrow it with ``match``).
        NotImplementedError: If the dataset is HDF5/ODIM radar (needs xradar).
        ImportError: If the optional 'grids'/'grib' dependencies are not installed.

    Example::

        import foehn

        # NetCDF: a coherent single-parameter slice of a multi-file collection
        ds = foehn.open_dataset("surface_derived_grid", match="rhiresd")
        ds = foehn.open_dataset("climate_scenarios_grid", match="_pr_", variables="pr")

        # GRIB2: a single forecast field — variable + member + reference + lead time
        ds = foehn.open_dataset("forecast_icon_ch1", match="202605231500-0-t_2m-ctrl")
    """
    _validate_grid_dataset(dataset)
    fmt = COLLECTION_META[dataset]["format"]
    reader = _GRID_READERS[fmt]

    xr = _require_xarray()
    if fmt == "GRIB2":
        _require_cfgrib()
        if match is None:
            raise ValueError(
                f"Dataset {dataset!r} is a GRIB2 forecast collection of thousands of single-field "
                "files; opening it unfiltered would download them all. Narrow to one field with "
                "match= (variable, member, reference + lead time), e.g. "
                f'foehn.open_dataset({dataset!r}, match="202605231500-0-t_2m-ctrl").'
            )

    data_dir = Path(data_dir) if data_dir else Path.cwd() / "data" / "meteoswiss"
    bronze_dir = data_dir / "bronze"
    files = _ensure_grid_files(
        dataset, bronze_dir, suffixes=reader["suffixes"], match=match, max_files=reader["max_files"]
    )

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

    if variables is not None:
        wanted = [variables] if isinstance(variables, str) else list(variables)
        ds = ds[wanted]

    return ds


def _store_slug(match: str) -> str:
    """Filesystem-safe fragment derived from a ``match`` filter for store names."""
    return re.sub(r"[^0-9A-Za-z]+", "_", match).strip("_") or "match"


def to_zarr(
    dataset: str,
    *,
    variables: str | list[str] | None = None,
    match: str | None = None,
    data_dir: Path | str | None = None,
    store: Path | str | None = None,
    rechunk: dict[str, int] | None = None,
    mode: str = "w",
) -> Path:
    """Materialise a gridded dataset to a Zarr store on disk.

    The grid analog of ``foehn.to_parquet()``: reads the source NetCDF or GRIB2
    via ``open_dataset()`` and writes a single Zarr store under ``data_dir/zarr/``.
    (GRIB2 collections require ``match`` — see ``open_dataset``.)

    The default store name encodes ``match`` so that different filtered slices of
    the same collection don't silently overwrite each other:
    ``<dataset>.zarr`` when unfiltered, ``<dataset>__<match>.zarr`` otherwise
    (e.g. ``surface_derived_grid__rhiresd.zarr``). Pass ``store`` for an explicit
    path that overrides this.

    Args:
        dataset: Dataset name. Must be a NetCDF or GRIB2 collection.
        variables: Restrict to these data variable(s) before writing.
        match: Narrow a multi-file collection to a coherent set (see open_dataset);
            required for GRIB2 collections.
        data_dir: Root data directory. Defaults to ./data/meteoswiss.
        store: Explicit output path for the ``.zarr`` store. Overrides the
            derived ``data_dir/zarr/<name>.zarr`` location when given.
        rechunk: Optional dim→chunk-size mapping applied before writing, e.g.
            ``{"time": 24}``. Requires ``dask`` (not part of the 'grids' extra —
            install separately with ``pip install dask``); raises ImportError
            if it is missing.
        mode: Zarr write mode (default "w" — overwrite the store at this path).
            Note distinct ``match`` values map to distinct default paths, so this
            only overwrites a prior run of the *same* slice, not a different one.

    Returns:
        Path to the written ``.zarr`` store.
    """
    ds = open_dataset(dataset, variables=variables, match=match, data_dir=data_dir)
    ds = _sanitize_noncf_time_units(ds)

    if store is not None:
        store = Path(store)
    else:
        data_dir = Path(data_dir) if data_dir else Path.cwd() / "data" / "meteoswiss"
        name = dataset if match is None else f"{dataset}__{_store_slug(match)}"
        store = data_dir / "zarr" / f"{name}.zarr"
    store.parent.mkdir(parents=True, exist_ok=True)

    if rechunk:
        import importlib.util

        if importlib.util.find_spec("dask") is None:
            raise ImportError(
                "to_zarr(rechunk=...) requires dask, which is not part of the "
                "'grids' extra. Install it with:\n\n  pip install dask\n"
            )
        ds = ds.chunk(rechunk)

    with warnings.catch_warnings():
        # xarray writes consolidated metadata by default. zarr-python 3 warns
        # that this is outside the Zarr v3 spec, but we keep it on purpose: it
        # is purely additive (every per-array zarr.json is still written, so
        # readers that ignore it still work), it makes the common open_zarr()
        # path fast and warning-free, and zarr-python reads it back natively.
        # Suppress only that specific, deliberate warning — nothing else.
        warnings.filterwarnings(
            "ignore",
            message="Consolidated metadata is currently not part in the Zarr format 3 specification",
        )
        ds.to_zarr(store, mode=mode)
    return store
