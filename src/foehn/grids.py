"""How one gridded :class:`~foehn.collections.DatasetKind` becomes an xarray Dataset.

The grid counterpart to :mod:`foehn.readers`, and placed the same way: below
:mod:`foehn.registry`, so :class:`~foehn.registry.KindSpec` can carry a
:class:`GridReader` beside its ``download``, ``convert`` and ``load`` adapters.
The public ``open_dataset`` and ``to_zarr`` live in :mod:`foehn.api`, exactly as
``load`` does.

Upstream's file conventions are not here: ODIM's gain/offset/nodata scaling is
:mod:`foehn.odim`, ICON's unstructured cell coordinates are :mod:`foehn.icon`,
and getting a match's files onto disk is :mod:`foehn.gridfiles`. What is left is
the readers themselves — the same split the tabular path has between
``readers``, ``meteocsv``, ``assets`` and ``transfer``.

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

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from foehn import icon, odim
from foehn.collections import COLLECTION_META
from foehn.fetch import Fetcher
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


def require_dask() -> None:
    """A :data:`Require` for ``rechunk=``: dask is *not* part of the 'grids' extra.

    Named here with the other three so every optional-import message foehn can
    raise sits in one place, even though nothing in ``KINDS`` carries this one —
    rechunking is a caller's request, not a property of a kind.
    """
    import importlib.util

    if importlib.util.find_spec("dask") is None:
        raise ImportError(
            "to_zarr(rechunk=...) requires dask, which is not part of the "
            "'grids' extra. Install it with:\n\n  pip install dask\n"
        )


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
    return icon.attach_lonlat(
        ds,
        dataset,
        workspace,
        fetcher=fetcher,
        what="returning the unstructured grid without coordinates.",
    )


def open_radar(files: list[Path], **_: object) -> xr.Dataset:
    """Open one ODIM-H5 Cartesian composite."""
    xr = _require_xarray()
    return odim.open_composite(xr, files[0])


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
        ds = odim.open_composite(xr, path)
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
    cube = icon.attach_lonlat(
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
    "open_grib2",
    "open_netcdf",
    "open_radar",
    "require_dask",
    "require_grib2",
    "require_netcdf",
    "require_radar",
    "sanitize_noncf_time_units",
    "select_variables",
    "write_zarr",
]
