"""How one gridded :class:`~foehn.collections.DatasetKind` becomes an xarray Dataset.

The grid counterpart to :mod:`foehn.readers`, and placed the same way: below
:mod:`foehn.registry`, so :class:`~foehn.registry.KindSpec` can carry a
:class:`GridReader` beside its ``download``, ``convert`` and ``load`` adapters.
The public ``open_dataset`` and ``to_zarr`` live in :mod:`foehn.api`, exactly as
``load`` does.

Upstream's file conventions are not here: ODIM's gain/offset/nodata scaling is
:mod:`foehn.odim`, ICON's unstructured cell coordinates are :mod:`foehn.icon`,
and getting a match's files onto disk is an injected acquisition adapter. What
is left is the readers themselves — the same split the tabular path has between
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

import contextlib
import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from foehn import atomicwrite, icon, odim
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


def _require_dask() -> None:
    """dask is *not* part of the 'grids' extra, and ``rechunk=`` needs it.

    Named here with the other three so every optional-import message foehn can
    raise sits in one place. Private, unlike them: nothing in ``KINDS`` carries
    it — rechunking is a caller's request, not a property of a kind — so it is
    checked inside :func:`write_zarr`, where the rechunk happens.
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
        engine: str | None = None,
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


class AcquireAdapter(Protocol):
    """Materialize the grid files selected by one read request."""

    def __call__(
        self,
        dataset: str,
        workspace: Workspace,
        *,
        suffixes: tuple[str, ...],
        match: str | None,
        max_files: int | None,
        run_datetime: bool,
        fetcher: Fetcher,
    ) -> list[Path]: ...


@dataclass(frozen=True)
class GridReader:
    """How one grid kind is read. Constructed in :data:`~foehn.registry.KINDS`.

    The per-kind functions live here beside the implementation they call; the
    configuration is stated in the registry row, exactly as ``key_columns`` and
    ``sort_column`` are for the tabular readers.
    """

    acquire: AcquireAdapter
    """Injected file-acquisition boundary; implemented above this read layer."""

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

    def open_dataset(
        self,
        dataset: str,
        *,
        match: str | None,
        variables: str | list[str] | None,
        workspace: Workspace,
        fetcher: Fetcher,
        engine: str | None = None,
    ) -> xr.Dataset:
        """Validate, materialize, and open one Grid Dataset."""
        if self.max_files == 1 and match is None:
            fmt = COLLECTION_META[dataset]["format"]
            raise ValueError(
                f"Dataset {dataset!r} is a {fmt} collection of many single-field files; opening it "
                "unfiltered would download them all. Narrow to one file with match=, e.g. "
                f'foehn.open_dataset({dataset!r}, match="{self.match_example}").'
            )
        self.require()
        files = self.acquire(
            dataset,
            workspace,
            suffixes=self.suffixes,
            match=match,
            max_files=self.max_files,
            run_datetime=self.run_datetime,
            fetcher=fetcher,
        )
        return select_variables(
            self.open(files, dataset=dataset, workspace=workspace, fetcher=fetcher, engine=engine), variables
        )

    def write_store(
        self,
        dataset: str,
        store: Path,
        *,
        match: str | None,
        variables: str | list[str] | None,
        rechunk: dict[str, int] | None,
        mode: str,
        stack: bool,
        workspace: Workspace,
        fetcher: Fetcher,
    ) -> None:
        """Write a Zarr store behind the Grid reader seam."""
        if stack and rechunk:
            raise ValueError("rechunk= is not supported with stack= (the cube is written separately).")

        if stack and self.cube is not None:
            if match is None:
                raise ValueError(
                    f'stack= needs match= to scope the cube for {dataset!r}, e.g. match="{self.cube_match_example}".'
                )
            self.require()
            files = self.acquire(
                dataset,
                workspace,
                suffixes=self.suffixes,
                match=match,
                max_files=self.cube_max_files,
                run_datetime=self.run_datetime,
                fetcher=fetcher,
            )
            if mode == "w":
                with atomicwrite.staged_directory(store) as staged:
                    self.cube(
                        files,
                        staged,
                        dataset=dataset,
                        workspace=workspace,
                        fetcher=fetcher,
                        variables=variables,
                        mode=mode,
                    )
            else:
                self.cube(
                    files,
                    store,
                    dataset=dataset,
                    workspace=workspace,
                    fetcher=fetcher,
                    variables=variables,
                    mode=mode,
                )
            return

        ds = self.open_dataset(
            dataset,
            match=match,
            variables=variables,
            workspace=workspace,
            fetcher=fetcher,
        )
        if mode == "w":
            with atomicwrite.staged_directory(store) as staged:
                _write_zarr(ds, staged, mode, rechunk=rechunk)
        else:
            _write_zarr(ds, store, mode, rechunk=rechunk)


# --- Writing ---------------------------------------------------------------


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


def _write_zarr(
    ds, store: Path, mode: str = "w", *, rechunk: dict[str, int] | None = None, append_dim: str | None = None
) -> None:
    """Write inside a staged replacement or directly into an append target.

    Everything a Dataset needs on its way to a store, so a caller learns one
    call rather than three. The non-CF time units have to be moved aside or the
    store throws on every later ``open_zarr``; ``rechunk`` needs dask, which the
    'grids' extra does not install. Both used to be the registry's to sequence —
    which meant the routing table knew the recipe, and ``cube_grib2`` stated
    half of it a second time on its own way out.
    """
    ds = _sanitize_noncf_time_units(ds)
    if rechunk:
        _require_dask()
        ds = ds.chunk(rechunk)
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


def write_zarr(
    ds, store: Path, mode: str = "w", *, rechunk: dict[str, int] | None = None, append_dim: str | None = None
) -> None:
    """Stage replacement stores; append to existing stores without copying them."""
    if mode == "w":
        with atomicwrite.staged_directory(store) as staged:
            _write_zarr(ds, staged, mode, rechunk=rechunk, append_dim=append_dim)
    else:
        _write_zarr(ds, store, mode, rechunk=rechunk, append_dim=append_dim)


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
        # Opened one at a time so a failure can close what is already open. The
        # comprehension this replaces leaked a handle per file whenever the
        # combine raised, which on a large set exhausts the descriptor limit and
        # reports the exhaustion instead of the original fault. On success the
        # handles stay open: the combined Dataset reads from them lazily.
        datasets: list = []
        try:
            for source in files:
                # Not a comprehension: the except below has to close whatever
                # was opened before the failure, which needs the partial list.
                datasets.append(xr.open_dataset(source, **kwargs))  # noqa: PERF401
            return xr.combine_by_coords(datasets, combine_attrs="drop_conflicts")
        except BaseException:
            for opened in datasets:
                with contextlib.suppress(Exception):
                    opened.close()
            raise

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


def open_netcdf(files: list[Path], *, dataset: str, engine: str | None = None, **_: object) -> xr.Dataset:
    """Open one or more NetCDF files, combining them on their coordinates.

    ``engine`` is passed to xarray, for a caller who needs to name a backend —
    ``h5netcdf`` where netCDF4 chokes on a particular file. Left unset, xarray
    picks. This was a documented keyword at v0.4.0 and its removal broke calls
    that named a backend, so it stays.
    """
    xr = _require_xarray()
    try:
        return _open_grid(xr, files, engine=engine)
    except OSError:
        # An unreadable file is not a heterogeneous set, and telling its owner
        # to narrow the match sends them somewhere the fix cannot be. The
        # underlying error already names the file that would not open, so it is
        # more use unchanged — a corrupt entry in the cache is deleted, not
        # matched around.
        raise
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


def _time_keys(ds) -> frozenset:
    """The timestamps *ds* carries, as integer nanoseconds.

    Normalised to one unit because the two sides of the comparison do not share
    one: an ODIM composite opens at ``datetime64[s]`` while the same instant read
    back out of a Zarr store is ``datetime64[ns]``. Worse, ``.tolist()`` on those
    two returns different *types* — ``datetime`` for seconds, ``int`` for
    nanoseconds — so the sets never intersected and every timestep looked new.
    """
    return frozenset(ds["time"].values.ravel().astype("datetime64[ns]").astype("int64").tolist())


def _cube_times(xr, store: Path) -> frozenset:
    """The timestamps an existing cube already holds, or nothing if it has none.

    A store that cannot be opened is treated as holding nothing: the append then
    behaves exactly as it did before this check existed, rather than failing on
    the way to a write that would have worked.
    """
    if not store.exists():
        return frozenset()
    try:
        with warnings.catch_warnings():
            # foehn writes consolidated metadata, but this may be reading a store
            # written by something that did not. Only the time coordinate is
            # wanted, so the slower path is fine and the advice is not for us.
            warnings.filterwarnings("ignore", message=".*consolidated metadata.*", category=RuntimeWarning)
            with xr.open_zarr(store) as existing:
                if "time" not in existing.coords:
                    return frozenset()
                return _time_keys(existing)
    except Exception as exc:  # any unreadable store means "unknown", not "fail"
        logger.debug("Could not read existing times from %s (%s); appending without de-duplication", store, exc)
        return frozenset()


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

    # An append is scoped by a ``match``, and a match spans everything upstream
    # has published under it — not just what arrived since the last append. So
    # the file list overlaps whatever the store already holds, and appending it
    # wholesale writes those timesteps a second time: append [00:00], then a
    # listing of [00:00, 00:05], and the cube reads [00:00, 00:00, 00:05].
    # Nothing downstream de-duplicates a Zarr append, so it is checked here.
    already_stored = _cube_times(xr, store) if mode == "a" else frozenset()

    written = 0
    # Radar filenames embed a zero-padded timestamp, so lexical order is chronological.
    for path in sorted(files):
        ds = odim.open_composite(xr, path)
        if "time" not in ds.coords:
            raise ValueError(f"{path.name}: no time coordinate — cannot stack along time.")
        # Narrowed per file rather than once at the end: only one timestep is ever
        # in memory. Same rule as everywhere else, hence the same call.
        ds = select_variables(ds, variables).expand_dims("time")
        ds["time"].encoding.update(_STACK_TIME_ENCODING)
        if already_stored and _time_keys(ds) <= already_stored:
            logger.debug("%s: already in %s, not appending again", path.name, store.name)
            continue
        # "w" creates the store on the first write of a replacement — and on an
        # append whose target does not exist yet, where "a" has nothing to
        # extend. Everything after the first write extends what it just made.
        write_mode = "a" if written or (mode == "a" and store.exists()) else "w"
        _write_zarr(ds, store, write_mode, append_dim="time" if write_mode == "a" else None)
        written += 1


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
    _write_zarr(cube, store, mode)


__all__ = [
    "AcquireAdapter",
    "CubeAdapter",
    "GridReader",
    "OpenAdapter",
    "Require",
    "cube_grib2",
    "cube_radar",
    "open_grib2",
    "open_netcdf",
    "open_radar",
    "require_grib2",
    "require_netcdf",
    "require_radar",
    "select_variables",
    "write_zarr",
]
