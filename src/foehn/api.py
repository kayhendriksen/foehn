"""Public Python API for foehn."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from foehn import registry
from foehn.collections import (
    DATASETS,
    DEFAULT_TIME_SLICE,
    GRANULARITY_LABELS,
    TIME_SLICE_LABELS,
    options,
)
from foehn.docstrings import renders
from foehn.fetch import DEFAULT_WORKERS, default_fetcher
from foehn.metadata import TABLES as metadata_tables
from foehn.metadata import fetch_table
from foehn.readers import Filters
from foehn.transfer import DownloadResult
from foehn.workspace import Workspace

if TYPE_CHECKING:
    import xarray as xr

# A vocabulary or a default named in a docstring is rendered from the table that
# defines it, never retyped beside it — the argument ``mcp_server`` already makes
# for its tool descriptions, and ``help(foehn.load)`` is the same kind of surface.
_VOCABULARY = {
    "granularities": options(GRANULARITY_LABELS),
    "time_slices": options(TIME_SLICE_LABELS),
    "default_time_slice": DEFAULT_TIME_SLICE,
    "default_workers": str(DEFAULT_WORKERS),
}


__all__ = [
    "METADATA_TABLES",
    "download",
    "inventory",
    "list_datasets",
    "load",
    "metadata",
    "open_dataset",
    "parameters",
    "stations",
    "to_parquet",
    "to_zarr",
]


def list_datasets() -> list[dict]:
    """Return metadata about all available datasets.

    Each dict has keys: ``dataset``, ``collection_id``, ``category``, ``subcategory``,
    ``description``, ``format``, ``frequencies``, ``time_slices``.
    """
    return [{"dataset": key, "collection_id": row.collection, **row.published()} for key, row in DATASETS.items()]


@renders(**_VOCABULARY)
def download(
    dataset: str,
    *,
    data_dir: Path | str | None = None,
    time_slice: list[str] | None = None,
    since: str | None = None,
    workers: int = DEFAULT_WORKERS,
    force: bool = False,
) -> DownloadResult:
    """Download a single dataset.

    Args:
        dataset: Dataset name (e.g. "smn"). Use list_datasets() to see options.
        data_dir: Root data directory. Defaults to ./data/meteoswiss.
        time_slice: Time slices to download. Options: $time_slices.
            Defaults to ["$default_time_slice"]. Ignored for binary/grid datasets
            (GRIB2/NetCDF), which fetch the latest assets.
        since: ISO timestamp for incremental updates — only assets updated after
            it are fetched. This function does not track state itself: pass the
            previous run's timestamp, or use the ``foehn download`` CLI, which
            persists one in ``_last_run.json`` and only advances it when the run
            fully succeeds.
        workers: Concurrent HTTP downloads (default $default_workers).
        force: Re-download even when local files look up to date. Currently
            only affects ZIP-shipped datasets (e.g. climate_scenarios_indoor),
            which otherwise skip when already extracted; other formats refresh
            via ``since``/ETags.

    Returns:
        DownloadResult summarising the download. Use ``result.downloaded > 0``
        to gate expensive downstream processing and ``result.failed`` to detect
        partial failures.
    """
    workspace = Workspace.resolve(data_dir)
    workspace.bronze().mkdir(parents=True, exist_ok=True)
    fetcher = default_fetcher()

    # Each kind knows its own download path, including the grid kinds, which have
    # no Parquet stage but still fetch their raw assets so the Python API mirrors
    # the CLI's --grids behaviour (and open_dataset/to_zarr).
    return registry.download(
        dataset,
        workspace,
        time_slice=time_slice,
        since=since,
        workers=workers,
        force=force,
        fetcher=fetcher,
    )


def to_parquet(
    dataset: str,
    *,
    data_dir: Path | str | None = None,
) -> None:
    """Convert downloaded CSVs to Parquet for a single dataset.

    Args:
        dataset: Dataset name (e.g. "smn").
        data_dir: Root data directory. Defaults to ./data/meteoswiss.

    Raises:
        RuntimeError: If one or more groups fail to convert. Each failure is
            also printed to stdout. Mirrors the CLI's exit-1 semantics so
            Python callers can't accidentally treat partial output as success.
    """
    failures = registry.convert(dataset, Workspace.resolve(data_dir))
    if failures:
        raise RuntimeError(
            f"to_parquet({dataset!r}) failed: {failures} group(s) did not convert. See stdout for details."
        )


# Re-exported: the tables and their published column names are ``foehn.metadata``'s,
# but ``METADATA_TABLES`` is a public name and the CLI builds its ``choices`` from it.
METADATA_TABLES = metadata_tables


def metadata(dataset: str, table: str) -> pl.DataFrame:
    """Fetch one of a dataset's metadata tables from the STAC API.

    The single implementation behind :func:`parameters`, :func:`stations` and
    :func:`inventory`; take it directly when the table is chosen at runtime, as
    the CLI does.

    Args:
        dataset: Dataset name (e.g. "smn"). Use list_datasets() to see options.
        table: One of ``METADATA_TABLES`` — "parameters", "stations", "inventory".

    Returns:
        A Polars DataFrame with foehn's published column names.
    """
    return fetch_table(dataset, table, fetcher=default_fetcher())


def parameters(dataset: str) -> pl.DataFrame:
    """Fetch parameter metadata for a dataset.

    Returns a DataFrame with columns: shortname, description, unit, type,
    granularity, decimals, group.

    Args:
        dataset: Dataset name (e.g. "smn"). Use list_datasets() to see options.
    """
    return metadata(dataset, "parameters")


def stations(dataset: str) -> pl.DataFrame:
    """Fetch station metadata for a dataset.

    Returns a DataFrame with columns: abbr, name, canton, altitude,
    lv95_east, lv95_north, lat, lon, data_since. ``data_since`` keeps the
    MeteoSwiss station metadata format (DD.MM.YYYY).

    Args:
        dataset: Dataset name (e.g. "smn"). Use list_datasets() to see options.
    """
    return metadata(dataset, "stations")


def inventory(dataset: str) -> pl.DataFrame:
    """Fetch the data inventory for a dataset.

    Returns a DataFrame with columns: station, parameter, data_since,
    data_till, owner.

    Args:
        dataset: Dataset name (e.g. "smn"). Use list_datasets() to see options.
    """
    return metadata(dataset, "inventory")


@renders(**_VOCABULARY)
def load(
    dataset: str,
    *,
    station: str | list[str] | None = None,
    frequency: str | list[str] | None = None,
    time_slice: str | list[str] | None = None,
    year: int | list[int] | None = None,
    month: int | list[int] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    columns: list[str] | None = None,
    drop_null: str | None = None,
    sort: str | None = None,
    limit: int | None = None,
    workers: int = DEFAULT_WORKERS,
) -> pl.DataFrame:
    """Load a dataset and return it as an in-memory Polars DataFrame.

    No files are written to disk. Data is fetched from the MeteoSwiss STAC API,
    parsed directly in memory, and returned as a single concatenated DataFrame.

    Args:
        dataset: Dataset name (e.g. "smn"). Use list_datasets() to see options.
        station: Station abbreviation(s) to include (e.g. "BER" or ["BER", "ZUR"]).
            Filters at the STAC item level so unmatched stations are never downloaded.
            Case-insensitive. If None, all stations are included.
        frequency: Granularity filter(s). Options: $granularities.
            Can be a single string or list. If None, all are included.
        time_slice: Time slice(s) to include. Options: $time_slices.
            Defaults to ["$default_time_slice"]. Can be a single string or list.
        year: Filter to specific year(s) (e.g. 2025 or [2020, 2021, 2022]).
        month: Filter to specific month(s) (1-12, e.g. 7 or [6, 7, 8] for summer).
        date_from: Start date (inclusive), ISO format "YYYY-MM-DD".
        date_to: End date (inclusive), ISO format "YYYY-MM-DD".
        columns: Only return these columns. ``station_abbr`` and
            ``reference_timestamp`` are always included.
        drop_null: Drop rows where this column is null.
        sort: Sort by timestamp. Options: "asc" (oldest first) or "desc"
            (newest first).
        limit: Cap the returned DataFrame to N rows. Applied after sort/columns.
            NOTE: this only bounds the returned shape — it does not reduce
            network bytes, since CSVs are per-station and not pre-bucketed by
            row count. To reduce wire volume, narrow with ``time_slice="now"``,
            ``year``, or ``date_from/date_to``.
        workers: Concurrent CSV downloads (default $default_workers). The CSV fetches are
            parallelised via a ThreadPoolExecutor; metadata fetch stays serial.

    Returns:
        A Polars DataFrame containing all matching CSV data.

    Example::

        import foehn

        # Recent daily data for Bern
        df = foehn.load("smn", station="BER", frequency="d")

        # Hourly data for multiple stations
        df = foehn.load("smn", station=["BER", "ZUR"], frequency="h")

        # Daily data for Bern, only January 2026
        df = foehn.load("smn", station="BER", frequency="d", year=2026, month=1)

        # Latest 10 readings (sort+limit are applied after the CSV is parsed)
        df = foehn.load("smn", station="BER", frequency="t",
                        time_slice="now", sort="desc", limit=10)

        # Summer 2025 temperatures, sorted newest first
        df = foehn.load("smn", station="BER", frequency="d",
                        time_slice="historical", date_from="2025-06-01",
                        date_to="2025-08-31", columns=["tre200d0"],
                        sort="desc")
    """
    filters = Filters.build(
        station=station,
        frequency=frequency,
        time_slice=time_slice,
        year=year,
        month=month,
        date_from=date_from,
        date_to=date_to,
        columns=columns,
        drop_null=drop_null,
        sort=sort,
        limit=limit,
        workers=workers,
    )
    # Whether this dataset can answer that query — known, tabular, granularity,
    # calendar filters, vocabulary — is every one of them a fact on its row, so
    # registry.load checks them rather than restating them here.
    return registry.load(dataset, filters, fetcher=default_fetcher())


# --- Gridded datasets --------------------------------------------------------


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
    use ``to_zarr(..., stack=True)`` instead.

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
        engine: xarray backend to open the file(s) with (e.g. "h5netcdf"). Leave
            unset to let xarray choose, which is right for every published
            collection; name one only to work around a backend-specific fault.

    Returns:
        An xarray Dataset backed by the local file(s), downloaded in full first
        (see the download-then-lazy note above) — e.g. the first
        ``climate_scenarios_grid`` call fetches ~900 MB before you read a pixel.

    Raises:
        ValueError: If the dataset is unknown, tabular (CSV), a GRIB2/radar
            collection opened without a single-file ``match``, if the match
            selects more files than the kind will open at once, or if its files
            cannot be combined into a single Dataset (narrow it with ``match``).
        OSError: If one of the matched files cannot be read — a corrupt entry in
            the local cache. The message names the file; delete it and retry.
        ImportError: If the optional 'grids' dependencies are not installed.
            Raised before anything is downloaded.

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
    return registry.open_grid(
        dataset,
        match=match,
        variables=variables,
        workspace=Workspace.resolve(data_dir),
        fetcher=default_fetcher(),
        engine=engine,
    )


def _stack_flag(stack: bool | str) -> bool:
    """Normalise ``stack=`` and reject a value that means nothing.

    v0.4.0 took ``"auto"``/``"time"`` and raised ValueError on anything else.
    Narrowing it to a bool made every other string truthy instead, so
    ``stack="bogus"`` quietly cubed rather than complaining. Both v0.4.0 tokens
    still resolve to True — the dataset's kind decides the method now.
    """
    if isinstance(stack, bool):
        return stack
    if stack in {"auto", "time"}:
        return True
    raise ValueError(
        f"stack={stack!r} is not a valid value. Pass a bool; the v0.4 tokens 'auto' and 'time' are "
        "also accepted and mean True (the dataset's kind decides how to cube)."
    )


def to_zarr(
    dataset: str,
    *,
    variables: str | list[str] | None = None,
    match: str | None = None,
    data_dir: Path | str | None = None,
    store: Path | str | None = None,
    rechunk: dict[str, int] | None = None,
    mode: str = "w",
    stack: bool | str = False,
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
            required for GRIB2 and radar collections, and for ``stack``.
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
            Replacement mode is staged; ``"a"`` writes in place so its work and
            temporary disk use scale with the new data, without rollback on interruption.
        stack: Assemble the matched files into one cube, using whichever method
            the dataset's kind uses — radar stacks CombiPrecip timesteps into a
            ``(time, y, x)`` cube incrementally (dask-free, one timestep in
            memory); GRIB2 promotes whichever of number/time/step vary into an
            N-D cube (e.g. ``(time, step, values)``) via ``combine_by_coords``
            (whole set in memory, capped at 1000 files). NetCDF has no cube
            builder — a multi-file ``match`` already combines on read — so
            ``stack`` is a no-op there. Incompatible with ``rechunk``.

    Returns:
        Path to the written ``.zarr`` store.
    """
    workspace = Workspace.resolve(data_dir)
    # Where the default store lands, and how ``match`` is spelled into its name,
    # are the workspace's layout rather than facts stated here.
    store_path = Path(store) if store is not None else workspace.zarr(dataset, match)
    store_path.parent.mkdir(parents=True, exist_ok=True)

    # Which method writes it — cube or single — is the kind's row, not a branch
    # here. So is whether the dataset has a Zarr path at all, which is what makes
    # this the unknown-dataset guard too.
    registry.write_zarr(
        dataset,
        store_path,
        match=match,
        variables=variables,
        rechunk=rechunk,
        mode=mode,
        stack=_stack_flag(stack),
        workspace=workspace,
        fetcher=default_fetcher(),
    )
    return store_path
