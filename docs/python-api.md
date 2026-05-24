# Python API

Use foehn directly from notebooks or scripts. All functions are available from the top-level `foehn` module.

---

## Listing datasets

```python
import foehn

foehn.list_datasets()
# [{'dataset': 'smn', 'collection_id': 'ch.meteoschweiz.ogd-smn', 'category': 'A',
#   'subcategory': 'A1', 'description': 'Automatic weather stations',
#   'format': 'CSV', 'frequencies': ['t', 'h', 'd', 'm', 'y'],
#   'time_slices': ['historical', 'recent', 'now']}, ...]
```

---

## Loading data

`foehn.load()` fetches data directly into a Polars DataFrame without writing anything to disk.

```python
# Load daily data for a single station
df = foehn.load("smn", station="BER", frequency="d")

# Multiple stations and frequencies
df = foehn.load("smn", station=["BER", "ZUR"], frequency=["d", "h"])

# Include historical data alongside recent
df = foehn.load("smn", station="BER", frequency="d", time_slice=["historical", "recent"])
```

### Filtering

All filters can be combined.

```python
# Filter by year and month
df = foehn.load("smn", station="BER", frequency="d", year=2026, month=[1, 2, 3])

# Date range
df = foehn.load("smn", station="BER", frequency="d",
                time_slice="historical", date_from="2025-06-01", date_to="2025-08-31")

# Select specific columns (station_abbr and reference_timestamp are always kept)
df = foehn.load("smn", station="BER", frequency="d", columns=["tre200d0", "rre150d0"])

# Drop rows where a column is null
df = foehn.load("obs", frequency="y", drop_null="w3pnd2y0")

# Sort by timestamp (newest first)
df = foehn.load("smn", station="BER", frequency="d", sort="desc")
```

| Parameter | Type | Description |
|---|---|---|
| `station` | `str` or `list[str]` | Station abbreviation(s) |
| `frequency` | `str` or `list[str]` | Time frequency: t, h, d, m, y |
| `time_slice` | `str` or `list[str]` | Time period: historical, recent, now |
| `year` | `int` or `list[int]` | Filter by year(s) |
| `month` | `int` or `list[int]` | Filter by month(s), 1--12 |
| `date_from` | `str` | Start date inclusive, ISO format |
| `date_to` | `str` | End date inclusive, ISO format |
| `columns` | `list[str]` | Only return these columns |
| `drop_null` | `str` | Drop rows where this column is null |
| `sort` | `str` | Sort by timestamp: "asc" or "desc" |

### Climate scenario collections

The CH2025 scenario collections have their own layout but load the same way:

```python
# Indoor climate scenarios (CSV+ZIP): the whole archive is fetched once, then
# filtered in memory by station.
df = foehn.load("climate_scenarios_indoor", station="ABO")

# Local climate scenarios (C8): a wide table with one column per climate model.
# Dates are a nominal 30-year period (0001-01-01 .. 0030-12-31), so the calendar
# filters (year/month/date_from/date_to) don't apply; `sort` orders by the
# string `date` column.
df = foehn.load("climate_scenarios", station="ABE")
```

---

## Metadata

Explore dataset metadata fetched live from the MeteoSwiss API.

```python
# Column name mappings: shortname, description, unit, type, ...
foehn.parameters("smn")

# Station info: abbreviation, name, canton, altitude, LV95 x/y, lat, lon, ...
foehn.stations("smn")

# What each station measures and since when
foehn.inventory("smn")
```

---

## Downloading to disk

```python
# Download a single dataset
foehn.download("smn", data_dir="./data/meteoswiss")

# Download with specific time slices
foehn.download("smn", time_slice=["historical", "recent"])
```

CSVs and metadata files are written to `<data_dir>/bronze/<collection>/`.

---

## Converting to Parquet

```python
foehn.to_parquet("smn", data_dir="./data/meteoswiss")
```

Parquet files are written to `<data_dir>/parquet/<collection>/`.

---

## Gridded data (NetCDF / GRIB2 / radar)

Gridded collections are N-dimensional fields, not tables, so they open as xarray
Datasets via `foehn.open_dataset()` — the grid analog of `load()`.

```python
# NetCDF climate grids/normals/scenarios — needs pip install "foehn[grids]"
ds = foehn.open_dataset("surface_derived_grid", match="rhiresd")
foehn.to_zarr("surface_derived_grid", match="rhiresd")

# GRIB2 forecasts (ICON-CH1/CH2, KENDA) — needs pip install "foehn[grib]".
# match= is required and must select a single file (variable, member,
# reference + lead time):
ds = foehn.open_dataset("forecast_icon_ch1", match="202605231500-0-t_2m-ctrl")

# HDF5/ODIM radar composites (CombiPrecip, hail) — needs pip install "foehn[radar]".
# match= is required and must select a single file (one ~5-min timestep):
ds = foehn.open_dataset("radar_precip", match="cpc2613000000")
```

This path is *download-then-lazy*: the source file is fetched in full to the
local cache before any read. See the [gridded data documentation](grids.md) for
`open_dataset`, `to_zarr`, `match`, the GRIB2/radar single-file requirement, and
the Swiss-grid coordinate notes.
