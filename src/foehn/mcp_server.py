"""MCP server for foehn — exposes MeteoSwiss data to LLM clients.

Provides mostly read-only access to Swiss meteorological open data through a set
of query tools, a reference guide resource, and a prompt template. The tabular
tools fetch live from the MeteoSwiss STAC API without touching local state; the
one exception is ``describe_grid``, which downloads the source grid file (NetCDF,
GRIB2, or radar) into the local bronze cache (annotated ``read_only_hint=False``).
"""

from __future__ import annotations

import logging
from typing import Literal

import polars as pl
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

import foehn
from foehn.collections import (
    COLLECTION_META,
    COLLECTIONS,
    GRIB2_COLLECTIONS,
    NETCDF_COLLECTIONS,
)

logger = logging.getLogger(__name__)

# Datasets that can be loaded as tabular data (CSV-backed).
_LOADABLE_DATASETS = sorted(k for k in COLLECTIONS if k not in GRIB2_COLLECTIONS and k not in NETCDF_COLLECTIONS)
_VALID_FREQUENCIES = {"t", "h", "d", "m", "y"}
_VALID_TIME_SLICES = {"historical", "recent", "now"}
_VALID_CATEGORIES = {"A", "C", "D", "E"}

# The tabular query tools are read-only against the MeteoSwiss API (describe_grid
# is the exception — it caches to disk, see _GRID_INSPECT below).
_READ_ONLY = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=True,
)

# describe_grid inspects a gridded collection (NetCDF, GRIB2, or radar). It is
# idempotent and never destructive, but it is not read-only: opening a grid
# downloads the source file in full to the local bronze cache (download-then-lazy).
_GRID_INSPECT = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=True,
)

# Gridded collections describe_grid can inspect: every NetCDF, GRIB2, and HDF5
# (radar) collection (GRIB2_COLLECTIONS holds both GRIB2 and radar). GRIB2/radar
# additionally require a single-file match=, enforced by open_dataset.
_INSPECTABLE_GRIDS = sorted(NETCDF_COLLECTIONS | GRIB2_COLLECTIONS)

mcp = MCPServer(
    "foehn",
    instructions=(
        "Powered by foehn (https://github.com/kayhendriksen/foehn) — "
        "the Python toolkit for Swiss meteorological data.\n\n"
        "Data provided by MeteoSwiss Open Government Data "
        "(https://github.com/MeteoSwiss/opendata).\n\n"
        "Typical workflow:\n"
        "1. list_datasets() — find a dataset (e.g. 'smn' for weather stations)\n"
        "2. get_stations(dataset) — find station abbreviations (e.g. 'BER' for Bern)\n"
        "3. get_parameters(dataset) — understand what each column measures\n"
        "4. describe_data(dataset, ...) — check row count and date range before loading\n"
        "5. load_data(dataset, ...) — fetch the data with filters\n\n"
        "Key tip: For historical data, always use time filters (year, month,\n"
        "date_from/date_to) to avoid hitting the row limit. Use drop_null to\n"
        "filter sparse datasets. Use sort='desc' to get newest data first.\n\n"
        "Gridded datasets (NetCDF climate grids, GRIB2 forecasts, radar) are not\n"
        "tabular — inspect them with describe_grid(dataset), not load_data();\n"
        "GRIB2 and radar need a single-file match=.\n\n"
        "Read the foehn://guide resource for detailed documentation.\n\n"
        "When presenting results, mention that they are powered by foehn "
        "with data from MeteoSwiss."
    ),
)


# ── Structured output models ───────────────────────────────────────────────


class Dataset(BaseModel):
    dataset: str = Field(description="Short name used in API calls (e.g. 'smn')")
    collection_id: str = Field(description="STAC collection ID")
    category: str = Field(description="MeteoSwiss category: A (ground), C (climate), D (radar), E (forecast)")
    subcategory: str = Field(description="Subcategory code (e.g. 'A1')")
    description: str = Field(description="Human-readable description")
    format: str = Field(description="Data format (CSV, GRIB2, NetCDF, etc.)")
    frequencies: list[str] = Field(description="Supported time frequencies (t, h, d, m, y)")
    time_slices: list[str] = Field(description="Available time slices (historical, recent, now)")


class Parameter(BaseModel):
    shortname: str = Field(description="Column name in data files (e.g. 'tre200s0')")
    description: str = Field(description="Human-readable description (e.g. 'Air temperature 2m above ground')")
    unit: str = Field(description="Measurement unit (e.g. '°C', 'mm', 'hPa')")
    type: str = Field(description="Data type")
    granularity: str = Field(description="Temporal granularity")
    decimals: int | str = Field(description="Number of decimal places")
    group: str = Field(description="Parameter group (e.g. 'Temperature', 'Precipitation')")


class Station(BaseModel):
    abbr: str = Field(description="Station abbreviation (e.g. 'BER' for Bern)")
    name: str = Field(description="Full station name")
    canton: str = Field(description="Swiss canton code (e.g. 'BE', 'ZH')")
    altitude: int | float = Field(description="Altitude in metres above sea level")
    lv95_east: float = Field(description="LV95 easting in metres (EPSG:2056)")
    lv95_north: float = Field(description="LV95 northing in metres (EPSG:2056)")
    lat: float = Field(description="WGS84 latitude")
    lon: float = Field(description="WGS84 longitude")
    data_since: str = Field(description="Date measurements started (DD.MM.YYYY, as published by MeteoSwiss)")


class InventoryEntry(BaseModel):
    station: str = Field(description="Station abbreviation")
    parameter: str = Field(description="Parameter shortname")
    data_since: str = Field(description="Start of available data")
    data_till: str = Field(description="End of available data")
    owner: str = Field(description="Data owner")


class ColumnSummary(BaseModel):
    name: str = Field(description="Column name")
    dtype: str = Field(description="Data type (e.g. 'Int64', 'Float64', 'Datetime')")
    non_null_count: int = Field(description="Number of non-null values")
    null_count: int = Field(description="Number of null/missing values")


class DataSummary(BaseModel):
    dataset: str = Field(description="Dataset name")
    total_rows: int = Field(description="Total number of rows matching the filters")
    stations: list[str] = Field(description="Station abbreviations found in the data")
    date_min: str | None = Field(description="Earliest timestamp in the data (ISO format)")
    date_max: str | None = Field(description="Latest timestamp in the data (ISO format)")
    columns: list[ColumnSummary] = Field(description="Column names, types, and null counts")


class GridVariable(BaseModel):
    name: str = Field(description="Data variable name (e.g. 'TabsD' for daily mean temperature)")
    dims: list[str] = Field(description="Dimension names in order (e.g. ['time', 'y', 'x'])")
    shape: list[int] = Field(description="Size of each dimension, matching `dims`")
    dtype: str = Field(description="Array data type (e.g. 'float32')")
    long_name: str | None = Field(description="Human-readable description from the file, if present")
    units: str | None = Field(description="Physical units from the file, if present")


class GridSummary(BaseModel):
    dataset: str = Field(description="Grid dataset name")
    dimensions: dict[str, int] = Field(description="Dimension name → size (e.g. {'time': 365, 'y': 640, 'x': 370})")
    coordinates: list[str] = Field(description="Coordinate variable names (e.g. ['time', 'x', 'y', 'lat', 'lon'])")
    data_variables: list[GridVariable] = Field(description="Data variables with their dims, shape, dtype, and metadata")
    attributes: dict[str, str] = Field(description="Selected global attributes (title, institution, source, ...)")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _load_and_filter(
    dataset: str,
    station: list[str] | None,
    frequency: str | None,
    time_slice: str | None,
    year: list[int] | None,
    month: list[int] | None,
    date_from: str | None,
    date_to: str | None,
    columns: list[str] | None,
    drop_null: str | None,
    sort: str | None,
    limit: int | None = None,
) -> pl.DataFrame:
    """Load a dataset and apply all filters via foehn.load()."""
    kwargs: dict = {}
    if station:
        kwargs["station"] = station
    if frequency:
        kwargs["frequency"] = frequency
    if time_slice:
        kwargs["time_slice"] = time_slice
    if year:
        kwargs["year"] = year
    if month:
        kwargs["month"] = month
    if date_from:
        kwargs["date_from"] = date_from
    if date_to:
        kwargs["date_to"] = date_to
    if columns:
        kwargs["columns"] = columns
    if drop_null:
        kwargs["drop_null"] = drop_null
    if sort:
        kwargs["sort"] = sort
    if limit is not None:
        kwargs["limit"] = limit
    return foehn.load(dataset, **kwargs)


# ── Tools ────────────────────────────────────────────────────────────────────


@mcp.tool(title="List datasets", annotations=_READ_ONLY)
def list_datasets(category: str | None = None) -> list[Dataset]:
    """List all available MeteoSwiss datasets.

    Returns dataset name, collection ID, category, description, format,
    supported frequencies, and available time slices for each dataset.

    Args:
        category: Filter by category. Options: "A" (ground-based measurements),
            "C" (climate), "D" (radar), "E" (forecasts). If omitted, returns all.
    """
    if category and category.upper() not in _VALID_CATEGORIES:
        raise ValueError(f"Invalid category {category!r}. Valid options: A, C, D, E.")

    datasets = foehn.list_datasets()
    if category:
        datasets = [d for d in datasets if d.get("category") == category.upper()]
    return [Dataset(**d) for d in datasets]


@mcp.tool(title="Load data", annotations=_READ_ONLY)
def load_data(
    dataset: str,
    station: list[str] | None = None,
    frequency: str | None = None,
    time_slice: str | None = None,
    year: list[int] | None = None,
    month: list[int] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    columns: list[str] | None = None,
    drop_null: str | None = None,
    sort: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Load weather measurements and return rows as a list of dicts.

    Fetches live data from MeteoSwiss. Only works with CSV-backed datasets
    (categories A, C1/C2, C8, E4). Binary/grid datasets (GRIB2, NetCDF)
    are not supported.

    **Tip:** For historical data, always combine time filters (year, month,
    date_from/date_to) to avoid hitting the row limit. Use describe_data()
    first if you're unsure about the data shape.

    Args:
        dataset: Dataset name (e.g. "smn" for SwissMetNet).
            Loadable datasets: smn, smn_precip, smn_tower, nime, tot, obs,
            pollen, phenology, nbcn, nbcn_precip, climate_scenarios, forecast_local.
        station: Station abbreviation(s) (e.g. ["BER"] for Bern, or ["BER", "ZUR"]).
            Case-insensitive. Use get_stations() to find abbreviations.
            If omitted, returns all stations (may be large).
        frequency: Time frequency. Options: "t" (10-min), "h" (hourly), "d" (daily),
            "m" (monthly), "y" (yearly). Not all datasets support all frequencies —
            check list_datasets() for the "frequencies" field.
        time_slice: Time period. Options: "historical" (start of measurements to end
            of last year), "recent" (this calendar year, default), "now" (last 24h).
        year: Filter to specific year(s) (e.g. [2025] or [2020, 2021, 2022]).
            Applied after loading, so use with time_slice="historical" to get
            past years without hitting the row limit.
        month: Filter to specific month(s) (1-12). E.g. [6, 7, 8] for summer.
            Can combine with year — year=[2025], month=[7] gives July 2025 only.
        date_from: Start date (inclusive), ISO format "YYYY-MM-DD" (e.g. "2025-06-01").
            More precise than year/month for arbitrary date ranges.
        date_to: End date (inclusive), ISO format "YYYY-MM-DD" (e.g. "2025-08-31").
        columns: Only return these columns (e.g. ["tre200s0", "rre150d0"]).
            station_abbr and reference_timestamp are always included.
            Use get_parameters() to find column names.
        drop_null: Drop rows where this column is null (e.g. "w3pnd2y0" to keep
            only stations that actually recorded hail). Useful for sparse datasets.
        sort: Sort by timestamp. Options: "asc" (oldest first, default), "desc"
            (newest first). Use "desc" with a limit to get the most recent data.
        limit: Maximum rows to return (default 50, max 500). Use filters to stay
            within limits on large datasets.
    """
    if dataset not in COLLECTIONS:
        raise ValueError(f"Unknown dataset {dataset!r}. Call list_datasets() to see options.")
    if dataset in GRIB2_COLLECTIONS or dataset in NETCDF_COLLECTIONS:
        fmt = COLLECTION_META[dataset]["format"]
        raise ValueError(f"Dataset {dataset!r} is {fmt} (binary/grid) and cannot be loaded as tabular data.")
    if frequency and frequency.lower() not in _VALID_FREQUENCIES:
        raise ValueError(f"Invalid frequency {frequency!r}. Valid options: t, h, d, m, y.")
    if time_slice and time_slice.lower() not in _VALID_TIME_SLICES:
        raise ValueError(f"Invalid time_slice {time_slice!r}. Valid options: historical, recent, now.")
    if sort and sort not in ("asc", "desc"):
        raise ValueError(f"Invalid sort {sort!r}. Valid options: asc, desc.")

    limit = min(max(1, limit), 500)

    df = _load_and_filter(
        dataset,
        station,
        frequency,
        time_slice,
        year,
        month,
        date_from,
        date_to,
        columns,
        drop_null,
        sort,
        limit=limit,
    )
    return df.to_dicts()


@mcp.tool(title="Describe data", annotations=_READ_ONLY)
def describe_data(
    dataset: str,
    station: list[str] | None = None,
    frequency: str | None = None,
    time_slice: str | None = None,
    year: list[int] | None = None,
    month: list[int] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    drop_null: str | None = None,
) -> DataSummary:
    """Get a summary of a dataset without returning actual rows.

    Returns total row count, station list, date range, and column info
    (names, types, null counts). Use this to understand the shape and
    scope of data before calling load_data() — especially useful for
    planning queries on large historical datasets.

    Accepts the same filters as load_data so you can check exactly how
    many rows a query would return before fetching the data.

    **Cost note:** this still fetches the matching CSVs in full (they are not
    pre-bucketed by row count) and summarises them in memory — it returns no
    rows, but it is *not* cheaper than load_data() in network/parse terms. Always
    narrow with the time filters (year, month, date_from/date_to) for historical
    slices, exactly as you would for load_data().

    Args:
        dataset: Dataset name (e.g. "smn"). Call list_datasets() to see options.
        station: Station abbreviation(s) to filter by.
        frequency: Time frequency filter ("t", "h", "d", "m", "y").
        time_slice: Time period ("historical", "recent", "now").
        year: Filter to specific year(s) (e.g. [2025]).
        month: Filter to specific month(s) (1-12).
        date_from: Start date (inclusive), ISO format "YYYY-MM-DD".
        date_to: End date (inclusive), ISO format "YYYY-MM-DD".
        drop_null: Only count rows where this column is not null.
    """
    if dataset not in COLLECTIONS:
        raise ValueError(f"Unknown dataset {dataset!r}. Call list_datasets() to see options.")
    if dataset in GRIB2_COLLECTIONS or dataset in NETCDF_COLLECTIONS:
        fmt = COLLECTION_META[dataset]["format"]
        raise ValueError(f"Dataset {dataset!r} is {fmt} (binary/grid) and cannot be described.")
    if frequency and frequency.lower() not in _VALID_FREQUENCIES:
        raise ValueError(f"Invalid frequency {frequency!r}. Valid options: t, h, d, m, y.")
    if time_slice and time_slice.lower() not in _VALID_TIME_SLICES:
        raise ValueError(f"Invalid time_slice {time_slice!r}. Valid options: historical, recent, now.")

    df = _load_and_filter(
        dataset,
        station,
        frequency,
        time_slice,
        year,
        month,
        date_from,
        date_to,
        columns=None,
        drop_null=drop_null,
        sort=None,
    )

    stations = sorted(df["station_abbr"].unique().to_list()) if "station_abbr" in df.columns else []

    date_min = None
    date_max = None
    if "reference_timestamp" in df.columns and len(df) > 0:
        ts = df["reference_timestamp"]
        date_min = str(ts.min())
        date_max = str(ts.max())

    col_summaries = []
    for col in df.columns:
        null_count = df[col].null_count()
        col_summaries.append(
            ColumnSummary(
                name=col,
                dtype=str(df[col].dtype),
                non_null_count=len(df) - null_count,
                null_count=null_count,
            )
        )

    return DataSummary(
        dataset=dataset,
        total_rows=len(df),
        stations=stations,
        date_min=date_min,
        date_max=date_max,
        columns=col_summaries,
    )


@mcp.tool(title="Describe grid", annotations=_GRID_INSPECT)
def describe_grid(
    dataset: str,
    match: str | None = None,
    variables: list[str] | None = None,
) -> GridSummary:
    """Inspect a gridded dataset's structure without returning array values.

    The grid analog of describe_data() for the CSV path. Returns the dataset's
    dimensions, coordinates, data variables (each with dims, shape, dtype,
    units, and long_name), and key global attributes — enough to understand the
    grid before deciding whether to pull it down locally.

    Works for any gridded collection: NetCDF (climate grids, normals, scenarios,
    hail hazard), GRIB2 forecasts (ICON-CH1/CH2, KENDA), and HDF5/ODIM radar
    (CombiPrecip, hail). CSV datasets should use describe_data()/load_data().

    **`match` is required for GRIB2 and radar** and must select a single file —
    those collections are thousands of files. e.g. match="202605231500-0-t_2m-ctrl"
    (forecast) or match="cpc2613000000" (radar). NetCDF can be inspected unfiltered
    or narrowed with match.

    **Cost warning:** foehn is download-then-lazy — the first call downloads the
    *entire* matched file(s) to the local cache before it can report anything
    (hundreds of MB for some NetCDF collections, ~900 MB for
    climate_scenarios_grid; a single GRIB2/radar file is only a few MB). Requires
    the optional 'grids' dependencies (pip install foehn[grids]).

    Args:
        dataset: Grid dataset name (e.g. "surface_derived_grid", "forecast_icon_ch1",
            "radar_precip"). Call list_datasets() to see options.
        match: Keep only source files whose name contains this substring. Required
            for GRIB2/radar (must select one file); narrows multi-file NetCDF
            collections (e.g. match="rhiresd" for daily precipitation).
        variables: Restrict the summary to these data variable(s).
    """
    if dataset not in COLLECTIONS:
        raise ValueError(f"Unknown dataset {dataset!r}. Call list_datasets() to see options.")
    if dataset not in _INSPECTABLE_GRIDS:
        raise ValueError(f"Dataset {dataset!r} is CSV/tabular. Use describe_data() or load_data() instead.")
    # open_dataset enforces the single-file match= requirement for GRIB2/radar.

    ds = foehn.open_dataset(dataset, match=match, variables=variables)
    try:
        dimensions = {str(k): int(v) for k, v in ds.sizes.items()}
        coordinates = sorted(str(c) for c in ds.coords)
        data_variables = [
            GridVariable(
                name=str(name),
                dims=[str(d) for d in da.dims],
                shape=[int(s) for s in da.shape],
                dtype=str(da.dtype),
                long_name=da.attrs.get("long_name"),
                units=da.attrs.get("units"),
            )
            for name, da in ds.data_vars.items()
        ]
        attributes = {str(k): str(v) for k, v in ds.attrs.items()}
    finally:
        ds.close()

    return GridSummary(
        dataset=dataset,
        dimensions=dimensions,
        coordinates=coordinates,
        data_variables=data_variables,
        attributes=attributes,
    )


@mcp.tool(title="Get parameters", annotations=_READ_ONLY)
def get_parameters(dataset: str) -> list[Parameter]:
    """Get parameter descriptions for a dataset.

    Returns what measurements are available (e.g. temperature, pressure,
    wind speed) with their short name, unit, data type, granularity, and group.

    Only works with CSV-backed datasets. Call this to understand what the
    column names in load_data() results mean.

    Args:
        dataset: Dataset name (e.g. "smn"). Call list_datasets() to see options.
    """
    if dataset not in COLLECTIONS:
        raise ValueError(f"Unknown dataset {dataset!r}. Call list_datasets() to see options.")

    return [Parameter(**row) for row in foehn.parameters(dataset).to_dicts()]


@mcp.tool(title="Get stations", annotations=_READ_ONLY)
def get_stations(dataset: str) -> list[Station]:
    """Get station metadata for a dataset.

    Returns station abbreviation, name, canton, altitude (m a.s.l.),
    LV95 coordinates, latitude, longitude, and the date measurements started
    in MeteoSwiss station metadata format (DD.MM.YYYY).

    Args:
        dataset: Dataset name (e.g. "smn"). Call list_datasets() to see options.
    """
    if dataset not in COLLECTIONS:
        raise ValueError(f"Unknown dataset {dataset!r}. Call list_datasets() to see options.")

    return [Station(**row) for row in foehn.stations(dataset).to_dicts()]


@mcp.tool(title="Get inventory", annotations=_READ_ONLY)
def get_inventory(dataset: str) -> list[InventoryEntry]:
    """Get the data inventory for a dataset.

    Shows which parameters are available at which stations, and the time
    range (data_since, data_till) of available data.

    Args:
        dataset: Dataset name (e.g. "smn"). Call list_datasets() to see options.
    """
    if dataset not in COLLECTIONS:
        raise ValueError(f"Unknown dataset {dataset!r}. Call list_datasets() to see options.")

    return [InventoryEntry(**row) for row in foehn.inventory(dataset).to_dicts()]


# ── Resource ─────────────────────────────────────────────────────────────────


@mcp.resource("foehn://guide")
def usage_guide() -> str:
    """Complete reference guide for querying MeteoSwiss data with foehn tools."""
    loadable_list = "\n".join(
        f"  - `{ds}` — {COLLECTION_META[ds]['description']} "
        f"(frequencies: {', '.join(COLLECTION_META[ds]['frequencies']) or 'n/a'}, "
        f"time slices: {', '.join(COLLECTION_META[ds]['time_slices']) or 'none'})"
        for ds in _LOADABLE_DATASETS
    )
    grid_list = "\n".join(
        f"  - `{ds}` — {COLLECTION_META[ds]['description']} ({COLLECTION_META[ds]['format']})"
        for ds in _INSPECTABLE_GRIDS
    )
    return f"""\
# foehn — MeteoSwiss Data Access Guide

Powered by foehn (https://github.com/kayhendriksen/foehn), the Python
toolkit for accessing Swiss meteorological data.

Data provided by MeteoSwiss Open Government Data
(https://github.com/MeteoSwiss/opendata).

## Recommended workflow

1. **list_datasets()** — browse available datasets and their metadata.
2. **get_stations(dataset)** — find station abbreviations for a dataset.
3. **get_parameters(dataset)** — understand column names, units, and groups.
4. **describe_data(dataset, ...)** — check row count and date range before loading.
5. **load_data(dataset, station, frequency, time_slice, year, columns, limit)** — fetch data.

Always call get_parameters() if you need to interpret column names — they
use MeteoSwiss shortcodes (e.g. `tre200s0` = 2m air temperature in Celsius).

## Loadable datasets (CSV-backed, usable with load_data)

{loadable_list}

## Gridded datasets (inspect with describe_grid — not load_data)

These are N-dimensional grids, not tables, so load_data() does not apply. Use
**describe_grid(dataset)** to see their dimensions, coordinates, and variables —
it covers NetCDF, GRIB2 forecasts, and radar. GRIB2 and radar need a single-file
`match` (e.g. `match="202605231500-0-t_2m-ctrl"` or `match="cpc2613000000"`);
NetCDF can be inspected unfiltered or narrowed with `match`. describe_grid is
download-then-lazy (the first call pulls the matched file(s) to a local cache).
To read or convert the data itself, use the Python API (`foehn.open_dataset` /
`foehn.to_zarr`) or the `foehn open` / `foehn to-zarr` CLI commands.

{grid_list}

## Frequencies

- `t` — 10-minute (near real-time measurements)
- `h` — hourly
- `d` — daily
- `m` — monthly
- `y` — yearly

Not all datasets support all frequencies. The `frequencies` field in
list_datasets() shows what each dataset supports.

Timestamp convention (all UTC):
- t, h: timestamp = END of interval (16:00 means 15:50:01-16:00:00)
- d, m, y: timestamp = START of interval (2023-06-01 means all of June)

## Time slices

- `now` — last ~24 hours, updated every 10 minutes (t, h only)
- `recent` — this calendar year through yesterday, updated daily (default)
- `historical` — start of measurements through Dec 31 of last year

## Filtering

load_data() and describe_data() support these filters (all combinable):

- **year** — e.g. year=[2025] or year=[2020, 2021, 2022]
- **month** — e.g. month=[6, 7, 8] for summer months (works across years)
- **date_from / date_to** — e.g. date_from="2025-06-01", date_to="2025-08-31"
- **columns** — e.g. columns=["tre200s0"] to return only temperature
- **drop_null** — e.g. drop_null="w3pnd2y0" to keep only rows with hail data
- **sort** — "asc" (oldest first, default) or "desc" (newest first)

## Limits and performance

load_data() returns at most 500 rows per call. To manage large datasets:
- **Always use time filters** (year, month, date_from/date_to)
- **Use columns** to reduce output size
- **Use drop_null** on sparse datasets to skip empty rows
- **Use sort="desc"** with limit to get the most recent data first
- Specify stations and use coarser frequencies when possible
- **Call describe_data() first** to check how many rows your query will return

## Common examples

**Current temperature in Bern:**
load_data("smn", station=["BER"], frequency="t", time_slice="now",
          sort="desc", limit=1)
→ look at `tre200s0` column (2m air temperature, Celsius)

**Daily rainfall in Geneva this year:**
load_data("smn", station=["GVE"], frequency="d", columns=["rre150d0"])

**Hail days across all stations in 2025 (skip stations with no data):**
load_data("obs", frequency="y", time_slice="historical", year=[2025],
          columns=["w3pnd2y0"], drop_null="w3pnd2y0", limit=500)

**Summer 2025 temperatures in Bern:**
load_data("smn", station=["BER"], frequency="d", time_slice="historical",
          date_from="2025-06-01", date_to="2025-08-31",
          columns=["tre200d0", "tre200dx", "tre200dn"])

**Monthly hail in all Julys since 2000:**
load_data("obs", frequency="m", time_slice="historical",
          year=[2000, 2001, ..., 2025], month=[7],
          columns=["w3pnd2m0"], drop_null="w3pnd2m0", limit=500)

**Last 50 readings for a station (newest first):**
load_data("smn", station=["BER"], frequency="t", sort="desc", limit=50)

**Check data shape before loading:**
describe_data("obs", frequency="m", time_slice="historical", year=[2025])
→ shows row count, stations, date range, and columns with null counts

**Stations in canton Zurich:**
get_stations("smn") → filter by canton == "ZH"

**Longest historical records:**
get_inventory("nbcn") → check data_since for each station

**Compare multiple stations:**
load_data("smn", station=["BER", "ZUR", "GVE"], frequency="d")

## Attribution

When presenting results, mention that they are powered by foehn
(https://github.com/kayhendriksen/foehn) with data from MeteoSwiss,
licensed under Swiss Open Government Data.
"""


# ── Prompt ───────────────────────────────────────────────────────────────────


@mcp.prompt()
def query_weather(question: str) -> str:
    """Answer a weather question about Switzerland using MeteoSwiss data.

    Args:
        question: The weather question (e.g. "What was the temperature in Bern last week?")
    """
    return f"""\
Answer this weather question using the foehn MeteoSwiss tools: "{question}"

Follow this workflow:
1. Read the foehn://guide resource for available datasets and parameters.
2. Call list_datasets() to identify the right dataset (usually "smn" for weather stations).
3. Call get_stations(dataset) to find the station abbreviation for any location mentioned.
4. Call get_parameters(dataset) to understand what columns are returned and their units.
5. Call describe_data() to check row count and plan your query.
6. Call load_data() with filters (year, month, date_from/date_to, columns,
   drop_null, sort) to fetch only the data you need.
7. Summarise the result clearly with station name, parameter, time period, units, and values.

Mention that results are powered by foehn with data from MeteoSwiss."""


# ── Entry point ──────────────────────────────────────────────────────────────


def run(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None:
    """Start the MCP server."""
    mcp.run(transport=transport)
