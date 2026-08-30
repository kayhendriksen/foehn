"""Public Python API for foehn."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from foehn import registry
from foehn.assets import collection_assets
from foehn.client import (
    DownloadResult,
)
from foehn.collections import (
    COLLECTION_META,
    COLLECTIONS,
    GRANULARITIES,
    TIME_SLICES,
)
from foehn.convert import (
    decode_meteoswiss_csv,
)
from foehn.fetch import DEFAULT_WORKERS, default_fetcher
from foehn.grids import sanitize_noncf_time_units, write_zarr
from foehn.readers import Filters, apply_time_filters

if TYPE_CHECKING:
    import xarray as xr

__all__ = [
    "download",
    "inventory",
    "list_datasets",
    "load",
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
    return [{"dataset": key, "collection_id": cid, **COLLECTION_META[key]} for key, cid in COLLECTIONS.items()]


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
        time_slice: Time slices to download. Defaults to ["recent"]. Ignored for
            binary/grid datasets (GRIB2/NetCDF), which fetch the latest assets.
        since: ISO timestamp for incremental updates — only assets updated after
            it are fetched. This function does not track state itself: pass the
            previous run's timestamp, or use the ``foehn download`` CLI, which
            persists one in ``_last_run.json`` and only advances it when the run
            fully succeeds.
        workers: Concurrent HTTP downloads (default 8).
        force: Re-download even when local files look up to date. Currently
            only affects ZIP-shipped datasets (e.g. climate_scenarios_indoor),
            which otherwise skip when already extracted; other formats refresh
            via ``since``/ETags.

    Returns:
        DownloadResult summarising the download. Use ``result.downloaded > 0``
        to gate expensive downstream processing and ``result.failed`` to detect
        partial failures.
    """
    if dataset not in COLLECTIONS:
        raise ValueError(f"Unknown dataset: {dataset!r}. Use list_datasets() to see available datasets.")

    data_dir = Path(data_dir) if data_dir else Path.cwd() / "data" / "meteoswiss"
    bronze_dir = data_dir / "bronze"
    bronze_dir.mkdir(parents=True, exist_ok=True)
    fetcher = default_fetcher()

    # Each kind knows its own download path, including the grid kinds, which have
    # no Parquet stage but still fetch their raw assets so the Python API mirrors
    # the CLI's --grids behaviour (and open_dataset/to_zarr).
    return registry.download(
        dataset,
        bronze_dir,
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
    if dataset not in COLLECTIONS:
        raise ValueError(f"Unknown dataset: {dataset!r}. Use list_datasets() to see available datasets.")

    data_dir = Path(data_dir) if data_dir else Path.cwd() / "data" / "meteoswiss"
    bronze_dir = data_dir / "bronze"
    parquet_dir = data_dir / "parquet"
    failures = registry.convert(dataset, bronze_dir, parquet_dir)
    if failures:
        raise RuntimeError(
            f"to_parquet({dataset!r}) failed: {failures} group(s) did not convert. See stdout for details."
        )


def _fetch_metadata_csv(dataset: str, suffix: str) -> pl.DataFrame:
    """Fetch a collection-level metadata CSV from the STAC API.

    Args:
        dataset: Dataset name (e.g. "smn").
        suffix: Metadata file suffix (e.g. "_meta_parameters").

    Returns:
        A Polars DataFrame with the parsed CSV contents.
    """
    if dataset not in COLLECTIONS:
        raise ValueError(f"Unknown dataset: {dataset!r}. Use list_datasets() to see available datasets.")

    collection_id = COLLECTIONS[dataset]
    fetcher = default_fetcher()

    coll = fetcher.collection(collection_id)
    for asset in collection_assets(coll, suffixes=(".csv",), contains=suffix):
        content = decode_meteoswiss_csv(fetcher.get(asset.href, timeout=60).body)
        return pl.read_csv(content.encode("utf-8"), separator=";")

    raise ValueError(f"No {suffix} metadata found for dataset {dataset!r}.")


def parameters(dataset: str) -> pl.DataFrame:
    """Fetch parameter metadata for a dataset.

    Returns a DataFrame with columns: shortname, description, unit, type,
    granularity, decimals, group.

    Args:
        dataset: Dataset name (e.g. "smn"). Use list_datasets() to see options.
    """
    df = _fetch_metadata_csv(dataset, "_meta_parameters")
    return df.select(
        pl.col("parameter_shortname").alias("shortname"),
        pl.col("parameter_description_en").alias("description"),
        pl.col("parameter_unit").alias("unit"),
        pl.col("parameter_datatype").alias("type"),
        pl.col("parameter_granularity").alias("granularity"),
        pl.col("parameter_decimals").alias("decimals"),
        pl.col("parameter_group_en").alias("group"),
    )


def stations(dataset: str) -> pl.DataFrame:
    """Fetch station metadata for a dataset.

    Returns a DataFrame with columns: abbr, name, canton, altitude,
    lv95_east, lv95_north, lat, lon, data_since. ``data_since`` keeps the
    MeteoSwiss station metadata format (DD.MM.YYYY).

    Args:
        dataset: Dataset name (e.g. "smn"). Use list_datasets() to see options.
    """
    df = _fetch_metadata_csv(dataset, "_meta_stations")
    return df.select(
        pl.col("station_abbr").alias("abbr"),
        pl.col("station_name").alias("name"),
        pl.col("station_canton").alias("canton"),
        pl.col("station_height_masl").alias("altitude"),
        pl.col("station_coordinates_lv95_east").alias("lv95_east"),
        pl.col("station_coordinates_lv95_north").alias("lv95_north"),
        pl.col("station_coordinates_wgs84_lat").alias("lat"),
        pl.col("station_coordinates_wgs84_lon").alias("lon"),
        pl.col("station_data_since").alias("data_since"),
    )


def inventory(dataset: str) -> pl.DataFrame:
    """Fetch the data inventory for a dataset.

    Returns a DataFrame with columns: station, parameter, data_since,
    data_till, owner.

    Args:
        dataset: Dataset name (e.g. "smn"). Use list_datasets() to see options.
    """
    df = _fetch_metadata_csv(dataset, "_meta_datainventory")
    return df.select(
        pl.col("station_abbr").alias("station"),
        pl.col("parameter_shortname").alias("parameter"),
        pl.col("data_since"),
        pl.col("data_till"),
        pl.col("owner"),
    )


def _require_columns(df: pl.DataFrame, names: list[str], label: str) -> None:
    """Raise ValueError naming any of *names* the loaded frame doesn't have.

    Silently ignoring an unknown column turns a mistyped MeteoSwiss shortcode
    (``tre200dO`` for ``tre200d0``) into a plausible-looking wrong answer: the
    ``columns`` filter returns only the always-kept key columns, and
    ``drop_null`` keeps every null row it was asked to remove. Both are worth an
    error, especially on the MCP surface where the caller is an LLM guessing
    parameter names.
    """
    missing = [n for n in names if n not in df.columns]
    if not missing:
        return
    available = ", ".join(sorted(df.columns))
    raise ValueError(f"Unknown column(s) {missing} in {label}=. This dataset has: {available}")


def _apply_post_filters(df: pl.DataFrame, dataset: str, filters: Filters) -> pl.DataFrame:
    """Apply the row/column filters that are the same for every kind.

    Which columns an explicit ``columns=`` selection always keeps is a property of
    the dataset's kind, so it is read from the registry rather than passed in —
    callers were previously stating their own schema across this seam, and only
    one of the three got it right for its kind. Running this once here, after
    whichever reader produced the frame, is what keeps the readers free of it.
    """
    spec = registry.spec(dataset)
    df = apply_time_filters(df, filters)
    if filters.drop_null:
        _require_columns(df, [filters.drop_null], "drop_null")
        df = df.filter(pl.col(filters.drop_null).is_not_null())
    if filters.sort in ("asc", "desc"):
        df = df.sort(spec.sort_column, descending=(filters.sort == "desc"))
    if filters.columns:
        _require_columns(df, list(filters.columns), "columns")
        keep = [c for c in spec.key_columns if c in df.columns]
        keep += [c for c in filters.columns if c not in keep]
        df = df.select(keep)
    if filters.limit is not None:
        df = df.head(filters.limit)
    return df


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
        frequency: Time frequency filter(s). Options: "t" (10-min), "h" (hourly),
            "d" (daily), "m" (monthly), "y" (yearly). Can be a single string or list.
            If None, all frequencies are included.
        time_slice: Time slice(s) to include. Defaults to ["recent"].
            Options: "historical", "recent", "now". Can be a single string or list.
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
        workers: Concurrent CSV downloads (default 8). The CSV fetches are
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
    if dataset not in COLLECTIONS:
        raise ValueError(f"Unknown dataset: {dataset!r}. Use list_datasets() to see available datasets.")
    spec = registry.spec(dataset)
    if not spec.tabular:
        raise ValueError(registry.unreadable_message(dataset))
    if frequency is not None and not spec.supports_granularity:
        raise ValueError(f"Dataset {dataset!r} does not support frequency filtering.")
    if sort is not None and sort not in ("asc", "desc"):
        raise ValueError(f"Invalid sort {sort!r}. Valid options: asc, desc.")

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
    # Reject a token outside the vocabulary rather than quietly matching no
    # assets and reporting "no CSV files found" — a mistyped frequency is a
    # caller error, and the MCP layer used to catch it with its own copy of
    # these two sets.
    if filters.granularities and (unknown := sorted(filters.granularities - GRANULARITIES)):
        raise ValueError(f"Invalid frequency {unknown}. Valid options: {', '.join(sorted(GRANULARITIES))}.")
    if unknown_slices := sorted(set(filters.time_slices) - TIME_SLICES):
        raise ValueError(f"Invalid time_slice {unknown_slices}. Valid options: {', '.join(sorted(TIME_SLICES))}.")
    if not spec.supports_calendar_filters and filters.has_calendar_filter:
        raise ValueError(
            f"Dataset {dataset!r} uses nominal 30-year dates (0001..0030); "
            "year/month/date_from/date_to filters are not supported."
        )

    df = registry.load(dataset, filters, fetcher=default_fetcher())
    return _apply_post_filters(df, dataset, filters)


# --- Gridded datasets --------------------------------------------------------


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


def open_dataset(
    dataset: str,
    *,
    variables: str | list[str] | None = None,
    match: str | None = None,
    data_dir: Path | str | None = None,
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

    Returns:
        An xarray Dataset backed by the local file(s), downloaded in full first
        (see the download-then-lazy note above) — e.g. the first
        ``climate_scenarios_grid`` call fetches ~900 MB before you read a pixel.

    Raises:
        ValueError: If the dataset is unknown, tabular (CSV), a GRIB2/radar
            collection opened without a single-file ``match``, or if its files
            cannot be combined into a single Dataset (narrow it with ``match``).
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
    if dataset not in COLLECTIONS:
        raise ValueError(f"Unknown dataset: {dataset!r}. Use list_datasets() to see available datasets.")

    data_dir = Path(data_dir) if data_dir else Path.cwd() / "data" / "meteoswiss"
    return registry.open_grid(
        dataset,
        match=match,
        variables=variables,
        bronze_dir=data_dir / "bronze",
        fetcher=default_fetcher(),
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
    stack: bool = False,
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
    if dataset not in COLLECTIONS:
        raise ValueError(f"Unknown dataset: {dataset!r}. Use list_datasets() to see available datasets.")
    if stack and rechunk:
        raise ValueError("rechunk= is not supported with stack= (the cube is written separately).")

    data_dir = Path(data_dir) if data_dir else Path.cwd() / "data" / "meteoswiss"
    store_path = _resolve_store(dataset, match, data_dir, store)
    store_path.parent.mkdir(parents=True, exist_ok=True)

    # Whether this kind has a cube builder is a fact on its row. NetCDF has none
    # and needs none, so it falls through to the single-write path below.
    grid = registry.spec(dataset).grid
    if stack and grid is not None and grid.cube is not None:
        registry.write_cube(
            dataset,
            store_path,
            match=match,
            variables=variables,
            mode=mode,
            bronze_dir=data_dir / "bronze",
            fetcher=default_fetcher(),
        )
        return store_path

    ds = sanitize_noncf_time_units(open_dataset(dataset, variables=variables, match=match, data_dir=data_dir))

    if rechunk:
        import importlib.util

        if importlib.util.find_spec("dask") is None:
            raise ImportError(
                "to_zarr(rechunk=...) requires dask, which is not part of the "
                "'grids' extra. Install it with:\n\n  pip install dask\n"
            )
        ds = ds.chunk(rechunk)

    write_zarr(ds, store_path, mode)
    return store_path
