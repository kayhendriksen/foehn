"""MCP server for foehn — exposes MeteoSwiss data to LLM clients.

Provides read-only access to Swiss meteorological open data through five
tools, a reference guide resource, and a prompt template. All data is
fetched live from the MeteoSwiss STAC API; no local state is modified.
"""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP
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

# All tools are read-only queries against the MeteoSwiss API.
_READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=True,
)

mcp = FastMCP(
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
        "4. load_data(dataset, station, frequency, time_slice) — fetch the data\n\n"
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
    lat: float = Field(description="WGS84 latitude")
    lon: float = Field(description="WGS84 longitude")
    data_since: str = Field(description="Date measurements started (YYYY-MM-DD)")


class InventoryEntry(BaseModel):
    station: str = Field(description="Station abbreviation")
    parameter: str = Field(description="Parameter shortname")
    data_since: str = Field(description="Start of available data")
    data_till: str = Field(description="End of available data")
    owner: str = Field(description="Data owner")


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
    limit: int = 50,
) -> list[dict]:
    """Load weather measurements and return rows as a list of dicts.

    Fetches live data from MeteoSwiss. Only works with CSV-backed datasets
    (categories A, C1/C2, C8, E4). Binary/grid datasets (GRIB2, NetCDF)
    are not supported.

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
        limit: Maximum rows to return (default 50, max 500). Use a station filter
            or coarser frequency to stay within limits on large datasets.
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

    limit = min(max(1, limit), 500)

    kwargs: dict = {}
    if station:
        kwargs["station"] = station
    if frequency:
        kwargs["frequency"] = frequency
    if time_slice:
        kwargs["time_slice"] = time_slice

    df = foehn.load(dataset, **kwargs)
    return df.head(limit).to_dicts()


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
    latitude, longitude, and the date measurements started.

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
    binary_list = "\n".join(
        f"  - `{ds}` — {COLLECTION_META[ds]['description']} ({COLLECTION_META[ds]['format']})"
        for ds in sorted(GRIB2_COLLECTIONS | NETCDF_COLLECTIONS)
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
4. **load_data(dataset, station, frequency, time_slice, limit)** — fetch data.

Always call get_parameters() if you need to interpret column names — they
use MeteoSwiss shortcodes (e.g. `tre200s0` = 2m air temperature in Celsius).

## Loadable datasets (CSV-backed, usable with load_data)

{loadable_list}

## Binary/grid datasets (metadata only — cannot be loaded with load_data)

{binary_list}

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

## Limits and performance

load_data() returns at most 500 rows per call. To manage large datasets:
- Always specify a station when possible
- Use coarser frequencies (d instead of t)
- Use "recent" or "now" instead of "historical" for shorter time ranges

## Common examples

**Current temperature in Bern:**
load_data("smn", station=["BER"], frequency="t", time_slice="now", limit=1)
→ look at `tre200s0` column (2m air temperature, Celsius)

**Daily rainfall in Geneva this year:**
load_data("smn", station=["GVE"], frequency="d")
→ look at `rre150d0` column (daily precipitation sum, mm)

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
5. Call load_data(dataset, station, frequency, time_slice) to fetch the data.
6. Summarise the result clearly with station name, parameter, time period, units, and values.

Mention that results are powered by foehn with data from MeteoSwiss."""


# ── Entry point ──────────────────────────────────────────────────────────────


def run(transport: str = "stdio") -> None:
    """Start the MCP server."""
    mcp.run(transport=transport)
